import onnxruntime
import numpy as np
from transformers import AutoConfig, AutoProcessor, GenerationConfig
import os
from huggingface_hub import snapshot_download

# 1. Load models
## Load config and processor
model_id = "onnx-community/gemma-4-E2B-it-ONNX"
processor = AutoProcessor.from_pretrained(model_id)
config = AutoConfig.from_pretrained(model_id)
generation_config = GenerationConfig.from_pretrained(model_id)

## Load sessions
audio_model = "onnx/audio_encoder_q4.onnx"
vision_model = "onnx/vision_encoder_q4.onnx"
embed_model = "onnx/embed_tokens_q4.onnx"
decoder_model = "onnx/decoder_model_merged_q4.onnx"
model_dir = snapshot_download(model_id, allow_patterns=[f"{audio_model}*",  f"{vision_model}*", f"{embed_model}*", f"{decoder_model}*"])
audio_model_path   = os.path.join(model_dir, audio_model)
vision_model_path  = os.path.join(model_dir, vision_model)
embed_model_path   = os.path.join(model_dir, embed_model)
decoder_model_path = os.path.join(model_dir, decoder_model)

providers = ['CPUExecutionProvider']
vision_session  = onnxruntime.InferenceSession(vision_model_path, providers=providers)
audio_session   = onnxruntime.InferenceSession(audio_model_path, providers=providers)
embed_session   = onnxruntime.InferenceSession(embed_model_path, providers=providers)
decoder_session = onnxruntime.InferenceSession(decoder_model_path, providers=providers)
 
## Set config values
eos_token_id = generation_config.eos_token_id
image_token_id = config.image_token_id
audio_token_id = config.audio_token_id

# 2. Prepare inputs
## Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "In detail, describe the following audio and image."},
            {"type": "audio", "audio": "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav"},
            {"type": "image", "image": "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/artemis.jpeg"},
        ],
    },
]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
input_ids = inputs["input_ids"].numpy()
attention_mask = inputs["attention_mask"].numpy()
position_ids = np.cumsum(attention_mask, axis=-1) - 1

pixel_values = inputs["pixel_values"].numpy() if "pixel_values" in inputs else None
pixel_position_ids = inputs["image_position_ids"].numpy() if "image_position_ids" in inputs else None
input_features = inputs["input_features"].numpy().astype(np.float32) if "input_features" in inputs else None
input_features_mask = inputs["input_features_mask"].numpy() if "input_features_mask" in inputs else None

## Prepare decoder inputs
batch_size = input_ids.shape[0]
num_logits_to_keep = np.array(1, dtype=np.int64)
past_key_values = {
    inp.name: np.zeros(
        [batch_size, inp.shape[1], 0, inp.shape[3]],
        dtype=np.float32 if inp.type == "tensor(float)" else np.float16,
    )
    for inp in decoder_session.get_inputs()
    if inp.name.startswith("past_key_values")
}

# 3. Generation loop
max_new_tokens = 1024
generated_tokens = np.array([[]], dtype=np.int64)
image_features = None
audio_features = None
for i in range(max_new_tokens):
    inputs_embeds, per_layer_inputs = embed_session.run(None, {"input_ids": input_ids})
    if image_features is None and pixel_values is not None:
        image_features = vision_session.run(["image_features"], {"pixel_values": pixel_values, "pixel_position_ids": pixel_position_ids})[0]
        mask = (input_ids == image_token_id).reshape(-1)
        flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
        flat_embeds[mask] = image_features
        inputs_embeds = flat_embeds.reshape(inputs_embeds.shape)

    if audio_features is None and input_features is not None and input_features_mask is not None:
        audio_features = audio_session.run(
            ["audio_features"],
            {"input_features": input_features, "input_features_mask": input_features_mask},
        )[0]
        mask = (input_ids == audio_token_id).reshape(-1)
        flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
        flat_embeds[mask] = audio_features
        inputs_embeds = flat_embeds.reshape(inputs_embeds.shape)

    logits, *present_key_values = decoder_session.run(None, dict(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        per_layer_inputs=per_layer_inputs,
        position_ids=position_ids,
        num_logits_to_keep=num_logits_to_keep,
        **past_key_values,
    ))

    ## Update values for next generation loop
    input_ids = logits[:, -1].argmax(-1, keepdims=True)
    attention_mask = np.concatenate([attention_mask, np.ones_like(input_ids)], axis=-1)
    position_ids = position_ids[:, -1:] + 1
    for j, key in enumerate(past_key_values):
        past_key_values[key] = present_key_values[j]

    generated_tokens = np.concatenate([generated_tokens, input_ids], axis=-1)
    if np.isin(input_ids, eos_token_id).any():
        break

    ## (Optional) Streaming
    print(processor.decode(input_ids[0]), end="", flush=True)
print()

# 4. Output result
print(processor.batch_decode(generated_tokens, skip_special_tokens=True)[0])
