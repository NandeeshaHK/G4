"""
Gemma 4 Multimodal Server.

FastAPI server with:
- Warm model loading at startup (no cold-start delay)
- Ollama-compatible /api/chat endpoint
- Multimodal support (text, image, audio, video)
- SSE streaming with graceful error handling
- Temporary file management for uploads
"""

import os
import json
import uuid
import base64
import traceback

# ---------------------------------------------------------------------------
# Server Configuration (Read from Environment Variables)
# Defaults to True if not explicitly disabled
# ---------------------------------------------------------------------------
ENABLE_VISION = os.getenv("GEMMA_ENABLE_VISION", "true").lower() == "true"
ENABLE_AUDIO = os.getenv("GEMMA_ENABLE_AUDIO", "true").lower() == "true"
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import aiofiles
from scripts.gemma_engine import GemmaEngine
from scripts.video_utils import extract_frames


# ---------------------------------------------------------------------------
# Global Engine (loaded once at startup, kept warm)
# ---------------------------------------------------------------------------
engine: GemmaEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup and keep it warm."""
    global engine
    print(f"[Server] Loading Gemma 4 (Vision Enabled: {ENABLE_VISION}, Audio Enabled: {ENABLE_AUDIO})...")
    model_path = os.path.join(os.path.dirname(__file__), "models", "gemma-4-E2B-it-ONNX")
    if not os.path.exists(model_path):
        model_path = "./models/gemma-4-E2B-it-ONNX"
    engine = GemmaEngine(
        model_path, 
        enable_vision=ENABLE_VISION, 
        enable_audio=ENABLE_AUDIO
    )
    
    print("[Server] Running dummy inference to optimize GPU graphs...")
    # Send a tiny text prompt to trigger memory allocation
    dummy_msg = [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
    for _ in engine.generate_stream(dummy_msg, max_new_tokens=1):
        pass 
        
    print("[Server] Model warm, optimized, and ready!")
    yield
    print("[Server] Shutting down.")


app = FastAPI(title="Gemma 4 Multimodal Server", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories
os.makedirs("static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Request Schema
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    model: str = "gemma-4-e2b"
    stream: bool = True
    max_tokens: int = 1024
    temperature: float = 1.0
    visual_token_budget: int = 280
    video_fps: float = 1.0
    enable_thinking: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save_base64_to_file(data_uri: str) -> str:
    """Decode a data URI and save to uploads/. Returns the file path."""
    header, encoded = data_uri.split(",", 1)
    ext = header.split("/")[1].split(";")[0]  # e.g. "png", "wav"
    raw = base64.b64decode(encoded)
    path = f"uploads/{uuid.uuid4()}.{ext}"
    with open(path, "wb") as f:
        f.write(raw)
    return path


def _cleanup_files(paths: List[str]) -> None:
    """Remove temporary files silently."""
    for p in paths:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass


def preprocess_messages(
    messages: List[Dict[str, Any]],
    video_fps: float = 1.0,
) -> tuple[List[Dict[str, Any]], List[str]]:
    """
    Walk through messages and convert base64 payloads to local file paths.
    Returns (processed_messages, list_of_temp_files).
    """
    processed: List[Dict[str, Any]] = []
    temp_files: List[str] = []

    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            processed.append(msg)
            continue

        new_content: List[Dict[str, Any]] = []
        for item in content:
            m_type = item.get("type", "text")

            if m_type in ("image", "audio", "video"):
                source = item.get(m_type) or item.get("url")

                if source and source.startswith("data:"):
                    try:
                        path = _save_base64_to_file(source)
                        temp_files.append(path)
                        new_content.append({"type": m_type, m_type: path})
                        print(f"[Server] Saved {m_type} → {path}")
                    except Exception as e:
                        print(f"[Server] Error saving {m_type}: {e}")
                        new_content.append(item)

                elif m_type == "video" and source and os.path.exists(source):
                    frames = extract_frames(source, fps=video_fps)
                    for frame in frames:
                        new_content.append({"type": "image", "image": frame})
                    print(f"[Server] Extracted {len(frames)} frames from video")

                else:
                    new_content.append(item)
            else:
                new_content.append(item)

        processed.append({"role": msg["role"], "content": new_content})

    return processed, temp_files


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def read_index():
    async with aiofiles.open("static/index.html", mode="r") as f:
        return await f.read()


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file and return its server-side path for use in chat messages."""
    ext = file.filename.split(".")[-1] if "." in file.filename else "bin"
    path = f"uploads/{uuid.uuid4()}.{ext}"
    content = await file.read()
    with open(path, "wb") as f:
        f.write(content)
    print(f"[Server] Uploaded file → {path} ({len(content)} bytes)")
    return JSONResponse({"path": path, "size": len(content)})


@app.post("/api/chat")
async def chat(request: ChatRequest):
    global engine
    if engine is None:
        return {"error": "Model not loaded yet. Please wait for startup."}

    # SAFETY CHECK: Prevent processing files if the modality is disabled
    for msg in request.messages:
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                m_type = item.get("type", "text")
                if m_type in ["image", "video"] and not ENABLE_VISION:
                    return {"model": request.model, "message": {"role": "assistant", "content": "[Server Error: Vision capabilities are currently disabled on this server instance.]"}, "done": True}
                if m_type == "audio" and not ENABLE_AUDIO:
                    return {"model": request.model, "message": {"role": "assistant", "content": "[Server Error: Audio capabilities are currently disabled on this server instance.]"}, "done": True}

    # Pre-process messages (save base64 → files, expand video → frames)
    processed_messages, temp_files = preprocess_messages(
        request.messages, request.video_fps
    )

    print(f"[Server] Chat request: stream={request.stream}, "
          f"thinking={request.enable_thinking}, files={len(temp_files)}")

    # -----------------------------------------------------------------------
    # Streaming response
    # -----------------------------------------------------------------------
    if request.stream:
        def event_generator():
            try:
                for token in engine.generate_stream(
                    processed_messages,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    visual_token_budget=request.visual_token_budget,
                    enable_thinking=request.enable_thinking,
                ):
                    chunk = {
                        "model": request.model,
                        "message": {"role": "assistant", "content": token},
                        "done": False,
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                yield "data: [DONE]\n\n"

            except Exception as e:
                traceback.print_exc()
                err = {"model": request.model,
                       "message": {"role": "assistant",
                                   "content": f"\n\n[Server Error: {e}]"},
                       "done": True}
                yield f"data: {json.dumps(err)}\n\n"

            finally:
                _cleanup_files(temp_files)

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # -----------------------------------------------------------------------
    # Non-streaming response
    # -----------------------------------------------------------------------
    try:
        full_text = ""
        for token in engine.generate_stream(
            processed_messages,
            max_new_tokens=request.max_tokens,
            enable_thinking=request.enable_thinking,
        ):
            full_text += token
        return {
            "model": request.model,
            "message": {"role": "assistant", "content": full_text},
            "done": True,
        }
    except Exception as e:
        traceback.print_exc()
        return {"model": request.model,
                "message": {"role": "assistant", "content": f"[Error: {e}]"},
                "done": True}
    finally:
        _cleanup_files(temp_files)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
