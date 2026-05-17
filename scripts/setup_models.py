import os
import shutil
from huggingface_hub import snapshot_download

model_id = "onnx-community/gemma-4-E2B-it-ONNX"
local_dir = "models/gemma-4-E2B-it-ONNX"

# Ensure local dir exists
os.makedirs(local_dir, exist_ok=True)

# Define patterns for ONNX files as per the README
patterns = ["onnx/audio_encoder_q4.onnx*", "onnx/vision_encoder_q4.onnx*", "onnx/embed_tokens_q4.onnx*", "onnx/decoder_model_merged_q4.onnx*", "*.json", "*.txt"]

print(f"Checking for models in {model_id}...")
# snapshot_download will check the cache first. By setting local_dir, it will symlink or copy them.
# However, user wants a copy in the local repo.
# We'll use local_dir_use_symlinks=False to force copying.
try:
    path = snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        allow_patterns=patterns,
        local_dir_use_symlinks=False
    )
    print(f"Models are ready in: {path}")
except Exception as e:
    print(f"Error downloading/copying models: {e}")
