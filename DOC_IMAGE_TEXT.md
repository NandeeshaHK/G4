# Image + Text Server Usage Guide

This document explains how to interact with the Gemma 4 E2B server when configured for **Image & Text** capabilities.

When you start the server restricting audio (e.g., `GEMMA_ENABLE_AUDIO=false`), the backend ignores the audio processing tensors totally, dropping resource consumption. Any API request passing `"type": "audio"` will gracefully return an initialization error instead of pushing processing into an invalid graph.

## 1. Starting the Server

To launch the server optimized entirely for Visual workflows:

### **Windows (PowerShell)**:
```powershell
$env:GEMMA_ENABLE_AUDIO="false"; uv run python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### **Linux/macOS**:
```bash
GEMMA_ENABLE_AUDIO=false uv run python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## 2. API Endpoint Definition

**Endpoint:** `POST /api/chat`
**Content-Type:** `application/json`

Images can be sent straight as base64-encoded strings `data:image/...`. The server logic converts them seamlessly to image variables internally before pushing them toward the processor and mapping them to the CUDA encoder graph. 

You can also send a **video** file path, and the server will extract frames directly based on the `video_fps` parameter flag over the file!

### Supported Visual Formats:
- **Images:** `.jpg`, `.jpeg`, `.png`, `.webp`
- **Video:** `.mp4`, `.avi` (Only if referenced via server filepath, base64 video is tricky due to size).

### Request Schema Options:
- `messages` (required): The conversation blocks matching standard spec.
- `visual_token_budget` (optional, default: `280`): Constrains patch density generation for processing grid optimization.
- `video_fps` (optional, default: `1.0`): Controls how many image frames are extracted for 1 second of `"type": "video"`.
- `enable_thinking` (optional, default: `false`): Empowers `<|think|>` protocol responses.

---

## 3. Request Payload Examples

To prompt the model with an image, you place multiple dictionaries defining `"type"` inside the `content` block:

### A. Raw Base64 Image Parsing
```json
{
  "model": "gemma-4-e2b",
  "stream": true,
  "visual_token_budget": 500,
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text", 
          "text": "What is depicted in this image?"
        },
        {
          "type": "image", 
          "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        }
      ]
    }
  ]
}
```

### B. Referencing a Pre-Uploaded Video Endpoint
If handling gigantic visual tensors across requests is annoying, you can use the built-in multipart file uploader. 

```bash
POST /api/upload (multipart/form-data)
```

Then submit a Chat generation specifying the file:
```json
{
  "model": "gemma-4-e2b",
  "video_fps": 1.5,
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text", 
          "text": "Summarize the action happening in this video."
        },
        {
          "type": "video", 
          "video": "uploads/user-uploaded-vid.mp4"
        }
      ]
    }
  ]
}
```
