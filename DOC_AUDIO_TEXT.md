# Audio + Text Server Usage Guide

This document explains how to interact with the Gemma 4 E2B server when configured for **Audio & Text** interactions. 

When you start the server restricting vision (e.g., `GEMMA_ENABLE_VISION=false`), the server actively saves local RAM/VRAM arrays by skipping the vision session initialization altogether. Any API request that attempts to pass a video or image via `"type": "image"` will be intercepted and safely rejected to protect the pipeline.

## 1. Starting the Server

To launch the server specifically tailored for an Audio-enabled workflow:

### **Windows (PowerShell)**:
```powershell
$env:GEMMA_ENABLE_VISION="false"; uv run python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### **Linux/macOS**:
```bash
GEMMA_ENABLE_VISION=false uv run python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## 2. API Endpoint Definition

**Endpoint:** `POST /api/chat`
**Content-Type:** `application/json`

The backend handles base64-encoded audio directly. When it receives a base64 `data:audio/...` string, the server drops it to a temporary local volume, feeds the file directly into `librosa` behind the scenes, and passes it optimally into the audio processing layer cleanly. 

### Supported Audio Formats:
- `.wav`, `.mp3`, `.ogg`, `.flac`

### Request Schema Options:
- `messages` (required): The conversation list block matching the huggingface specification.
- `stream` (optional, default: `true`): Uses Server-Sent Events (SSE).
- `max_tokens` (optional, default: `1024`): The upper limit of completion tokens.
- `enable_thinking` (optional, default: `false`): Unlocks the reasoning context window utilizing `<|think|>` tokens.

---

## 3. Request Payload Example

To properly prompt the model with audio, specify an array in the `content` field assigning explicit `type` tags:

```json
{
  "model": "gemma-4-e2b",
  "stream": true,
  "enable_thinking": true,
  "max_tokens": 1024,
  "messages": [
    {
      "role": "system",
      "content": "You are a professional audio analysis AI."
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text", 
          "text": "Transcribe the following audio and tell me the speaker's tone:"
        },
        {
          "type": "audio", 
          "audio": "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA..."
        }
      ]
    }
  ]
}
```

## 4. Alternate Implementation: Pre-Upload
If the client cannot formulate massive base64 payloads easily, you can use the upload endpoint first.

**Step 1. Upload:**
```bash
POST /api/upload (multipart/form-data with "file" key)
Response: {"path": "uploads/1234.wav", "size": 102400}
```

**Step 2. Reference in Chat:**
```json
"content": [
  {"type": "text", "text": "Listen to this."},
  {"type": "audio", "audio": "uploads/1234.wav"}
]
```
