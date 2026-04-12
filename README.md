# Gemma 4 E2B Multimodal Server

This repository contains a high-performance multithreaded Fast API server acting as an endpoint for the `onnx-community/gemma-4-E2B-it-ONNX` multimodal model. It utilizes the powerful underlying native C++ arrays via Open Neural Network Exchange (ONNX) sessions, completely running out-of-core calculations and fully mapping text, image, and audio models across the system GPU properly via the `CUDAExecutionProvider`.

## 🚀 Features

* **Warm-Start Optimization:** Performs a headless inference at API lifecycle startup to allocate graphs explicitly onto the system VRAM to ensure that real user requests have 0 initial-turn loading logic overhead.
* **SSE Real-Time Streaming:** The API endpoint automatically spools back streamed partial-token Server-Sent Events natively onto separate Python threads, fully bypassing Python API async deadlock constraints.
* **Component-Level Modularity:** By utilizing Environment Variable switches, the engine can granularly opt in-and-out of booting components of its multimodal core to dynamically serve varied environments—freezing image features out, for example, frees massive amounts of local RAM overhead natively!
* **Gemma 4 "Thinking" Protocol Aware:** Safely injects the official `<|think|>` tokens into the user context before chat application parsing via flags.

## 🛠 Usage & Configuration

This server is highly configurable. You can launch instances optimized for specific server workloads directly via the command line, shutting down large portions of the machine learning model that are not necessary!

### Default Modality (Text + Image + Audio)
Start the server normally with all inference endpoints accessible and loaded:
```bash
uv run python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### Save 500MB VRAM: Audio + Text Only
Want to build an isolated speech agent while ignoring vision dependencies?
**For Mac/Linux:**
```bash
GEMMA_ENABLE_VISION=false uv run python -m uvicorn app:app --host 0.0.0.0 --port 8000
```
**For Windows (PowerShell):**
```powershell
$env:GEMMA_ENABLE_VISION="false"; uv run python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### Super-Lightweight Node: Text Only
Perfect for lightweight instances. Only `embed_tokens` and `decoder` nodes will be initialized onto the CUDA Provider.
**For Mac/Linux:**
```bash
GEMMA_ENABLE_VISION=false GEMMA_ENABLE_AUDIO=false uv run python -m uvicorn app:app --host 0.0.0.0 --port 8000
```
**For Windows (PowerShell):**
```powershell
$env:GEMMA_ENABLE_VISION="false"; $env:GEMMA_ENABLE_AUDIO="false"; uv run python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

## 🏗 Requirements
`onnxruntime-gpu`, `torch`, and native CUDA hooks must be properly configured across your UV environment for `CUDAExecutionProvider` parameters to capture natively.
