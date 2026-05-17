"""
Gemma 4 E2B ONNX Inference Engine.

Follows the official ONNX Runtime Python reference code pattern exactly:
- apply_chat_template handles ALL input processing (text, image, audio) in one call
- enable_thinking is passed directly to apply_chat_template
- Generation loop matches the reference implementation
"""

import os
import json
import gc
import sys
import threading
import traceback
import numpy as np
import soundfile as sf
import onnxruntime
from transformers import AutoProcessor, AutoConfig, GenerationConfig
from PIL import Image
from typing import List, Dict, Any, Optional, Generator

# Tell ONNX Runtime's CUDA BFC arena to shrink (release pages back to the
# driver) whenever an allocation is freed.  Without this the arena keeps
# every page it ever touched, causing idle VRAM to grow after each request.
os.environ.setdefault("ORT_CUDA_ARENA_SHRINK_WHEN_NOT_USED", "1")


class CancelledError(Exception):
    """Raised when generation is cancelled by a newer request."""
    pass


class GPUMemoryError(Exception):
    """Raised when ONNX Runtime runs out of GPU memory during inference."""
    pass


def _is_oom_error(exc: Exception) -> bool:
    """Check if an ONNX Runtime exception is a GPU memory allocation failure."""
    msg = str(exc)
    return "AllocateRawInternal" in msg or "Available memory of" in msg


def _provider_label(providers: List[Any]) -> str:
    """Return a printable provider name for logs."""
    if not providers:
        return "unknown"
    first = providers[0]
    return first if isinstance(first, str) else first[0]


def _slice_by_seq_axis(arr: Any, start: int, end: int, total_seq_len: int) -> Any:
    """Slice an array along whichever axis matches the token sequence length."""
    if not isinstance(arr, np.ndarray):
        return arr

    for axis, size in enumerate(arr.shape):
        if size == total_seq_len:
            slicer = [slice(None)] * arr.ndim
            slicer[axis] = slice(start, end)
            return arr[tuple(slicer)]

    return arr


class GemmaEngine:
    """Gemma 4 E2B multimodal inference engine using ONNX Runtime."""

    def __init__(self, model_dir: str, device: str = "cuda", enable_vision: bool = True, enable_audio: bool = True) -> None:
        self.model_dir = model_dir
        self.device = device
        self.model_id = "onnx-community/gemma-4-E2B-it-ONNX"

        # Save config state
        self.enable_vision = enable_vision
        self.enable_audio = enable_audio

        # Load Config and Processor
        print("[Engine] Loading processor and config...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.config = AutoConfig.from_pretrained(self.model_id)
        self.generation_config = GenerationConfig.from_pretrained(self.model_id)

        # Runtime tuning defaults (can be overridden by engine_config.json)
        self.prefill_chunk_size = 256
        self.gpu_mem_gb = 3.0
        self.device_map = self._load_device_map()

        # Base provider definitions
        self.cpu_providers = ["CPUExecutionProvider"]
        if device == "cuda":
            self.gpu_providers = [
                ("CUDAExecutionProvider", {
                    "device_id": 0,
                    "gpu_mem_limit": int(self.gpu_mem_gb * 1024 * 1024 * 1024),
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                }),
                "CPUExecutionProvider"
            ]
        else:
            self.gpu_providers = self.cpu_providers

        # Session options for performance
        sess_opts = onnxruntime.SessionOptions()
        sess_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = os.cpu_count()
        sess_opts.inter_op_num_threads = 2
        sess_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        # Disable memory pattern pre-allocation — it reserves GPU memory
        # at session load time based on peak usage from the first run,
        # which stays allocated even when idle.
        sess_opts.enable_mem_pattern = False
        # Disable internal buffer reuse so ORT frees intermediate tensors
        # immediately instead of keeping them for potential future reuse.
        sess_opts.enable_mem_reuse = False
        sess_opts.enable_cpu_mem_arena = True
        sess_opts.log_severity_level = 3  # suppress verbose warnings

        # Paths
        embed_path = os.path.join(model_dir, "onnx/embed_tokens_q4.onnx")
        decoder_path = os.path.join(model_dir, "onnx/decoder_model_merged_q4.onnx")

        text_providers = self._providers_for_modality("text")
        vision_providers = self._providers_for_modality("image")
        audio_providers = self._providers_for_modality("audio")

        print(
            "[Engine] Provider routing: "
            f"text={_provider_label(text_providers)}, "
            f"image={_provider_label(vision_providers)}, "
            f"audio={_provider_label(audio_providers)}"
        )
        print(
            "[Engine] Runtime tuning: "
            f"prefill_chunk_size={self.prefill_chunk_size}, "
            f"gpu_mem_gb={self.gpu_mem_gb}"
        )
        print(f"[Engine] Initializing Core Text ONNX sessions with: {_provider_label(text_providers)}")
        print(f"[Engine] Threads: intra={sess_opts.intra_op_num_threads}, inter={sess_opts.inter_op_num_threads}")
        
        self.embed_session = onnxruntime.InferenceSession(
            embed_path,
            sess_options=sess_opts,
            providers=text_providers,
        )
        self.decoder_session = onnxruntime.InferenceSession(
            decoder_path,
            sess_options=sess_opts,
            providers=text_providers,
        )

        # Conditionally load Vision Model
        if self.enable_vision:
            print(f"[Engine] Initializing Vision session with: {_provider_label(vision_providers)}")
            vision_path = os.path.join(model_dir, "onnx/vision_encoder_q4.onnx")
            self.vision_session = onnxruntime.InferenceSession(
                vision_path,
                sess_options=sess_opts,
                providers=vision_providers,
            )
        else:
            self.vision_session = None

        # Conditionally load Audio Model
        if self.enable_audio:
            print(f"[Engine] Initializing Audio session with: {_provider_label(audio_providers)}")
            audio_path = os.path.join(model_dir, "onnx/audio_encoder_q4.onnx")
            self.audio_session = onnxruntime.InferenceSession(
                audio_path,
                sess_options=sess_opts,
                providers=audio_providers,
            )
        else:
            self.audio_session = None

        # Constants
        self.eos_token_id = self.generation_config.eos_token_id
        self.image_token_id = self.config.image_token_id
        self.audio_token_id = self.config.audio_token_id

        print("[Engine] Model loaded and warm. Ready for inference.")

    def _load_device_map(self) -> Dict[str, str]:
        """Load per-modality device map from engine_config.json.

        Config format:
            {
                "audio": "cpu",
                "image": "gpu",
                "text": "cpu",
                "prefill_chunk_size": 256,
                "gpu_mem_gb": 4
            }

        Returns a dict mapping each modality to "cpu" or "gpu".
        Missing keys default to "gpu" when device is cuda.
        """
        default_device = "gpu" if self.device == "cuda" else "cpu"
        defaults = {"text": default_device, "image": default_device, "audio": default_device}

        config_paths: List[str] = []
        env_config = os.getenv("GEMMA_ENGINE_CONFIG", "").strip()
        if env_config:
            config_paths.append(env_config)

        model_dir_abs = os.path.abspath(self.model_dir)
        project_root = os.path.dirname(os.path.dirname(model_dir_abs))
        config_paths.append(os.path.join(project_root, "engine_config.json"))
        config_paths.append(os.path.join(model_dir_abs, "engine_config.json"))

        for path in config_paths:
            if not os.path.isfile(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception as exc:
                print(f"[Engine] Failed to parse config at {path}: {exc}")
                continue

            device_map = dict(defaults)
            valid_devices = {"cpu", "gpu"}
            for key in ("text", "image", "audio"):
                val = payload.get(key, default_device)
                if isinstance(val, str) and val.strip().lower() in valid_devices:
                    device_map[key] = val.strip().lower()
                else:
                    print(f"[Engine] Invalid device for '{key}': {val!r}. Using '{default_device}'.")

            raw_chunk = payload.get("prefill_chunk_size", self.prefill_chunk_size)
            if isinstance(raw_chunk, int) and raw_chunk > 0:
                self.prefill_chunk_size = raw_chunk
            else:
                print(
                    f"[Engine] Invalid prefill_chunk_size: {raw_chunk!r}. "
                    f"Using default {self.prefill_chunk_size}."
                )

            raw_gpu_mem = payload.get("gpu_mem_gb", self.gpu_mem_gb)
            if isinstance(raw_gpu_mem, (int, float)) and float(raw_gpu_mem) > 0:
                self.gpu_mem_gb = float(raw_gpu_mem)
            else:
                print(
                    f"[Engine] Invalid gpu_mem_gb: {raw_gpu_mem!r}. "
                    f"Using default {self.gpu_mem_gb}."
                )

            print(f"[Engine] Device map loaded from {path}: {device_map}")
            return device_map

        print("[Engine] No engine_config.json found. Using built-in runtime defaults.")
        return defaults

    def _providers_for_modality(self, modality: str) -> List[Any]:
        """Return providers for a specific modality based on device map."""
        if self.device != "cuda":
            return self.cpu_providers
        if self.device_map.get(modality, "gpu") == "cpu":
            return self.cpu_providers
        return self.gpu_providers

    def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        visual_token_budget: int = 280,
        enable_thinking: bool = False,
        max_context_tokens: int = 8192,
        cancel_event: Optional[threading.Event] = None,
    ) -> Generator[str, None, None]:
        """
        Stream tokens from the model following the official reference code.

        Key difference from previous implementation:
        - Uses apply_chat_template with tokenize=True, return_dict=True, return_tensors="pt"
          to process ALL inputs (text, images, audio) in a single call.
        - This ensures perfect alignment between image/audio tokens and their features.
        """
        print(f"[Engine] generate_stream called: {len(messages)} messages, "
              f"thinking={enable_thinking}, max_tokens={max_new_tokens}, "
              f"budget={visual_token_budget}")

        # =====================================================================
        # Step 1: Normalize messages so content is always a list of dicts.
        #         apply_chat_template expects: [{"type": "text", "text": "..."}, ...]
        # =====================================================================
        normalized: List[Dict[str, Any]] = []
        for msg in messages:
            c = msg.get("content", "")
            if isinstance(c, str):
                # Plain string → wrap in list format
                normalized.append({
                    "role": msg["role"],
                    "content": [{"type": "text", "text": c}] if c else []
                })
            else:
                normalized.append(msg)

        # =====================================================================
        # Step 1.5: Inject Official Thinking Control Token
        # =====================================================================
        if enable_thinking:
            # Check if the first message is already a system prompt
            if normalized and normalized[0]["role"] == "system":
                # Prepend the <|think|> token to the existing system text
                existing_text = normalized[0]["content"][0]["text"]
                normalized[0]["content"][0]["text"] = f"<|think|>{existing_text}"
            else:
                # Create a new system prompt consisting of just the token
                normalized.insert(0, {
                    "role": "system",
                    "content": [{"type": "text", "text": "<|think|>"}]
                })

        # =====================================================================
        # Step 1.6: Pre-load audio files and convert stereo → mono.
        #           apply_chat_template's internal loader doesn't handle
        #           multi-channel audio, causing a broadcast error in np.pad.
        # =====================================================================
        for msg in normalized:
            if not isinstance(msg.get("content"), list):
                continue
            for item in msg["content"]:
                if item.get("type") != "audio":
                    continue
                audio_src = item.get("audio")
                if audio_src is None or isinstance(audio_src, np.ndarray):
                    continue
                if isinstance(audio_src, str) and os.path.isfile(audio_src):
                    try:
                        data, sr = sf.read(audio_src, dtype="float32", always_2d=False)
                        if data.ndim == 2:
                            data = data.mean(axis=1)
                            print(f"[Engine] Converted stereo audio to mono: {audio_src}")
                        item["audio"] = data
                    except Exception as e:
                        print(f"[Engine] WARNING: Could not pre-load audio {audio_src}: {e}")

        # =====================================================================
        # Step 2: Use apply_chat_template to process EVERYTHING in one call.
        #         This is exactly what the reference code does.
        # =====================================================================
        try:
            inputs = self.processor.apply_chat_template(
                normalized,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=enable_thinking,
            )
        except Exception as e:
            print(f"[Engine] ERROR in apply_chat_template: {e}")
            traceback.print_exc()
            yield f"\n\n[Error processing input: {e}]"
            return

        input_ids = inputs["input_ids"].numpy()
        attention_mask = inputs["attention_mask"].numpy()
        position_ids = np.cumsum(attention_mask, axis=-1) - 1

        pixel_values = inputs["pixel_values"].numpy() if "pixel_values" in inputs else None
        pixel_position_ids = inputs["image_position_ids"].numpy() if "image_position_ids" in inputs else None
        input_features = inputs["input_features"].numpy().astype(np.float32) if "input_features" in inputs else None
        input_features_mask = inputs["input_features_mask"].numpy() if "input_features_mask" in inputs else None

        print(f"[Engine] input_ids shape: {input_ids.shape}, "
              f"has_image: {pixel_values is not None}, "
              f"has_audio: {input_features is not None}")

        # Context length guard
        if max_context_tokens > 0 and input_ids.shape[1] > max_context_tokens:
            over = input_ids.shape[1] - max_context_tokens
            if pixel_values is not None or input_features is not None:
                print(f"[Engine] WARNING: Context ({input_ids.shape[1]} tokens) exceeds "
                      f"max_context_tokens={max_context_tokens}. Multimodal inputs present — "
                      f"truncation skipped to preserve token alignment.")
            else:
                print(f"[Engine] Context truncated: {input_ids.shape[1]} → "
                      f"{max_context_tokens} tokens ({over} dropped from left).")
                input_ids = input_ids[:, over:]
                attention_mask = attention_mask[:, over:]
                position_ids = np.cumsum(attention_mask, axis=-1) - 1

        # =====================================================================
        # Step 2: Prepare decoder KV-cache (empty at start)
        # =====================================================================
        batch_size = input_ids.shape[0]
        num_logits_to_keep = np.array(1, dtype=np.int64)
        past_key_values = {
            inp.name: np.zeros(
                [batch_size, inp.shape[1], 0, inp.shape[3]],
                dtype=np.float32 if inp.type == "tensor(float)" else np.float16,
            )
            for inp in self.decoder_session.get_inputs()
            if inp.name.startswith("past_key_values")
        }

        # =====================================================================
        # Step 3: Prompt prefill (optionally chunked) + generation loop
        # =====================================================================
        prompt_input_ids = input_ids
        prompt_attention_mask = attention_mask
        prompt_position_ids = position_ids

        image_features = None
        audio_features = None
        generated_count = 0
        i = -1

        try:
            # 1. Embed the full prompt once.
            inputs_embeds, per_layer_inputs = self.embed_session.run(
                None, {"input_ids": prompt_input_ids}
            )

            # 2. Vision encoding (prompt-time only)
            if self.enable_vision and pixel_values is not None:
                image_features = self.vision_session.run(
                    ["image_features"],
                    {"pixel_values": pixel_values, "pixel_position_ids": pixel_position_ids}
                )[0]
                mask = (prompt_input_ids == self.image_token_id).reshape(-1)
                flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
                flat_embeds[mask] = image_features
                inputs_embeds = flat_embeds.reshape(inputs_embeds.shape)
                print(f"[Engine] Vision encoded: {image_features.shape}, mask_count={mask.sum()}")

            # 3. Audio encoding (prompt-time only)
            if self.enable_audio and input_features is not None and input_features_mask is not None:
                audio_features = self.audio_session.run(
                    ["audio_features"],
                    {"input_features": input_features, "input_features_mask": input_features_mask},
                )[0]
                mask = (prompt_input_ids == self.audio_token_id).reshape(-1)
                flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
                flat_embeds[mask] = audio_features
                inputs_embeds = flat_embeds.reshape(inputs_embeds.shape)
                print(f"[Engine] Audio encoded: {audio_features.shape}, mask_count={mask.sum()}")

            # 4. Decoder prefill. Chunking caps peak memory on long prompts.
            prompt_len = prompt_input_ids.shape[1]
            chunk_size = max(1, int(self.prefill_chunk_size))
            if chunk_size < prompt_len:
                print(f"[Engine] Chunked prefill: prompt_tokens={prompt_len}, chunk_size={chunk_size}")

            logits = None
            for start in range(0, prompt_len, chunk_size):
                if cancel_event and cancel_event.is_set():
                    print("[Engine] Cancelled during prefill.")
                    raise CancelledError("Generation cancelled.")

                end = min(prompt_len, start + chunk_size)
                chunk_embeds = inputs_embeds[:, start:end, :]
                chunk_per_layer_inputs = _slice_by_seq_axis(per_layer_inputs, start, end, prompt_len)
                chunk_position_ids = prompt_position_ids[:, start:end]
                chunk_attention_mask = prompt_attention_mask[:, :end]

                logits, *present_key_values = self.decoder_session.run(None, dict(
                    inputs_embeds=chunk_embeds,
                    attention_mask=chunk_attention_mask,
                    per_layer_inputs=chunk_per_layer_inputs,
                    position_ids=chunk_position_ids,
                    num_logits_to_keep=num_logits_to_keep,
                    **past_key_values,
                ))

                for j, key in enumerate(past_key_values):
                    past_key_values[key] = present_key_values[j]

            if logits is None:
                raise RuntimeError("Decoder prefill produced no logits.")

            # Continue from the full prompt state.
            attention_mask = prompt_attention_mask
            position_ids = prompt_position_ids
            input_ids = prompt_input_ids[:, -1:]

            for i in range(max_new_tokens):
                if cancel_event and cancel_event.is_set():
                    print(f"[Engine] Cancelled at token {i}.")
                    raise CancelledError("Generation cancelled.")

                # Prefill already produced logits for token 0.
                if i > 0:
                    step_embeds, step_per_layer_inputs = self.embed_session.run(
                        None, {"input_ids": input_ids}
                    )

                    logits, *present_key_values = self.decoder_session.run(None, dict(
                        inputs_embeds=step_embeds,
                        attention_mask=attention_mask,
                        per_layer_inputs=step_per_layer_inputs,
                        position_ids=position_ids,
                        num_logits_to_keep=num_logits_to_keep,
                        **past_key_values,
                    ))

                    for j, key in enumerate(past_key_values):
                        past_key_values[key] = present_key_values[j]

                # 5. Sample or greedily decode based on temperature
                next_logits = logits[:, -1]
                if temperature <= 0.0:
                    input_ids = next_logits.argmax(-1, keepdims=True)
                else:
                    scaled = next_logits / temperature
                    scaled -= scaled.max(-1, keepdims=True)  # stable softmax
                    probs = np.exp(scaled)
                    probs /= probs.sum(-1, keepdims=True)
                    vocab_size = probs.shape[-1]
                    input_ids = np.array(
                        [
                            [np.random.choice(
                                vocab_size,
                                p=(lambda p: p / p.sum())(probs[b].astype(np.float64)),
                            )]
                            for b in range(probs.shape[0])
                        ],
                        dtype=np.int64,
                    )

                # 6. Check EOS
                if np.isin(input_ids, self.eos_token_id).any():
                    print(f"[Engine] EOS reached at token {i}")
                    break

                # 7. Stream token
                token_text = self.processor.decode(input_ids[0])
                yield token_text
                generated_count += 1

                # 8. Update state for next iteration
                attention_mask = np.concatenate(
                    [attention_mask, np.ones_like(input_ids)], axis=-1
                )
                position_ids = position_ids[:, -1:] + 1

        except CancelledError:
            # Expected — the request was superseded by a newer one.
            # Re-raise so the worker can send the cancelled sentinel.
            raise
        except Exception as e:
            stage = "prefill" if i < 0 else f"token {i}"
            if _is_oom_error(e):
                seq_len = attention_mask.shape[-1] if attention_mask is not None else "unknown"
                print(f"[Engine] GPU OUT OF MEMORY during {stage} (sequence length: {seq_len})")
                raise GPUMemoryError(
                    f"GPU out of memory during generation (sequence length: {seq_len}). "
                    f"The input is too long for the available VRAM. "
                    f"Try: shorter audio/images, fewer conversation turns, lower max_context_tokens, "
                    f"or a smaller prefill_chunk_size."
                ) from e
            print(f"[Engine] ERROR during {stage}: {e}")
            traceback.print_exc()
            yield f"\n\n[Generation error during {stage}: {e}]"
            return
        finally:
            # Drop large temporary references so repeated requests release memory promptly.
            inputs = None
            input_ids = None
            prompt_input_ids = None
            prompt_attention_mask = None
            prompt_position_ids = None
            pixel_values = None
            pixel_position_ids = None
            input_features = None
            input_features_mask = None
            image_features = None
            audio_features = None
            past_key_values = {}
            gc.collect()

        print(f"[Engine] Generation complete: {generated_count} tokens")


# Standalone test
if __name__ == "__main__":
    engine = GemmaEngine("models/gemma-4-E2B-it-ONNX")
    msgs = [{"role": "user", "content": [{"type": "text", "text": "Hello, who are you?"}]}]
    for token in engine.generate_stream(msgs):
        print(token, end="", flush=True)
    print()
