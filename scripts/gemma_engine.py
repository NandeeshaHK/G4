"""
Gemma 4 E2B ONNX Inference Engine.

Follows the official ONNX Runtime Python reference code pattern exactly:
- apply_chat_template handles ALL input processing (text, image, audio) in one call
- enable_thinking is passed directly to apply_chat_template
- Generation loop matches the reference implementation
"""

import os
import sys
import traceback
import numpy as np
import onnxruntime
from transformers import AutoProcessor, AutoConfig, GenerationConfig
from PIL import Image
from typing import List, Dict, Any, Optional, Generator


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

        # Providers
        if device == "cuda":
            self.providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'gpu_mem_limit': 3 * 1024 * 1024 * 1024,
                    'arena_extend_strategy': 'kSameAsRequested',
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }),
                'CPUExecutionProvider'
            ]
        else:
            self.providers = ['CPUExecutionProvider']

        # Session options for performance
        sess_opts = onnxruntime.SessionOptions()
        sess_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = os.cpu_count()
        sess_opts.inter_op_num_threads = 2
        sess_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        sess_opts.enable_mem_pattern = True
        sess_opts.enable_cpu_mem_arena = True
        sess_opts.log_severity_level = 3  # suppress verbose warnings

        # Paths
        embed_path = os.path.join(model_dir, "onnx/embed_tokens_q4.onnx")
        decoder_path = os.path.join(model_dir, "onnx/decoder_model_merged_q4.onnx")

        print(f"[Engine] Initializing Core Text ONNX sessions with: {self.providers[0] if isinstance(self.providers[0], str) else self.providers[0][0]}")
        print(f"[Engine] Threads: intra={sess_opts.intra_op_num_threads}, inter={sess_opts.inter_op_num_threads}")
        
        self.embed_session = onnxruntime.InferenceSession(embed_path, sess_options=sess_opts, providers=self.providers)
        self.decoder_session = onnxruntime.InferenceSession(decoder_path, sess_options=sess_opts, providers=self.providers)

        # Conditionally load Vision Model
        if self.enable_vision:
            print("[Engine] Initializing Vision session...")
            vision_path = os.path.join(model_dir, "onnx/vision_encoder_q4.onnx")
            self.vision_session = onnxruntime.InferenceSession(vision_path, sess_options=sess_opts, providers=self.providers)
        else:
            self.vision_session = None

        # Conditionally load Audio Model
        if self.enable_audio:
            print("[Engine] Initializing Audio session...")
            audio_path = os.path.join(model_dir, "onnx/audio_encoder_q4.onnx")
            self.audio_session = onnxruntime.InferenceSession(audio_path, sess_options=sess_opts, providers=self.providers)
        else:
            self.audio_session = None

        # Constants
        self.eos_token_id = self.generation_config.eos_token_id
        self.image_token_id = self.config.image_token_id
        self.audio_token_id = self.config.audio_token_id

        print("[Engine] Model loaded and warm. Ready for inference.")

    def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        visual_token_budget: int = 280,
        enable_thinking: bool = False,
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
        # Step 3: Generation loop (matches reference code exactly)
        # =====================================================================
        image_features = None
        audio_features = None

        for i in range(max_new_tokens):
            try:
                # 1. Embed tokens
                inputs_embeds, per_layer_inputs = self.embed_session.run(
                    None, {"input_ids": input_ids}
                )

                # 2. Vision encoding (first iteration only)
                if self.enable_vision and image_features is None and pixel_values is not None:
                    image_features = self.vision_session.run(
                        ["image_features"],
                        {"pixel_values": pixel_values, "pixel_position_ids": pixel_position_ids}
                    )[0]
                    mask = (input_ids == self.image_token_id).reshape(-1)
                    flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
                    flat_embeds[mask] = image_features
                    inputs_embeds = flat_embeds.reshape(inputs_embeds.shape)
                    print(f"[Engine] Vision encoded: {image_features.shape}, mask_count={mask.sum()}")

                # 3. Audio encoding (first iteration only)
                if self.enable_audio and audio_features is None and input_features is not None and input_features_mask is not None:
                    audio_features = self.audio_session.run(
                        ["audio_features"],
                        {"input_features": input_features, "input_features_mask": input_features_mask},
                    )[0]
                    mask = (input_ids == self.audio_token_id).reshape(-1)
                    flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
                    flat_embeds[mask] = audio_features
                    inputs_embeds = flat_embeds.reshape(inputs_embeds.shape)
                    print(f"[Engine] Audio encoded: {audio_features.shape}, mask_count={mask.sum()}")

                # 4. Decoder step
                logits, *present_key_values = self.decoder_session.run(None, dict(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    per_layer_inputs=per_layer_inputs,
                    position_ids=position_ids,
                    num_logits_to_keep=num_logits_to_keep,
                    **past_key_values,
                ))

                # 5. Greedy decode
                input_ids = logits[:, -1].argmax(-1, keepdims=True)

                # 6. Check EOS
                if np.isin(input_ids, self.eos_token_id).any():
                    print(f"[Engine] EOS reached at token {i}")
                    break

                # 7. Stream token
                token_text = self.processor.decode(input_ids[0])
                yield token_text

                # 8. Update state for next iteration
                attention_mask = np.concatenate(
                    [attention_mask, np.ones_like(input_ids)], axis=-1
                )
                position_ids = position_ids[:, -1:] + 1
                for j, key in enumerate(past_key_values):
                    past_key_values[key] = present_key_values[j]

            except Exception as e:
                print(f"[Engine] ERROR at token {i}: {e}")
                traceback.print_exc()
                yield f"\n\n[Generation error at token {i}: {e}]"
                return

        print(f"[Engine] Generation complete: {i + 1} tokens")


# Standalone test
if __name__ == "__main__":
    engine = GemmaEngine("models/gemma-4-E2B-it-ONNX")
    msgs = [{"role": "user", "content": [{"type": "text", "text": "Hello, who are you?"}]}]
    for token in engine.generate_stream(msgs):
        print(token, end="", flush=True)
    print()
