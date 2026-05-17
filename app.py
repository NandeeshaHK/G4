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
import re
import json
import time
import uuid
import base64
import queue
import threading
import asyncio
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
from typing import List, Dict, Any, Optional
import uvicorn
import aiofiles
from scripts.gemma_engine import GemmaEngine, GPUMemoryError, CancelledError
from scripts.video_utils import extract_frames


# ---------------------------------------------------------------------------
# Global Engine (loaded once at startup, kept warm)
# ---------------------------------------------------------------------------
engine: GemmaEngine | None = None


# ---------------------------------------------------------------------------
# Inference Worker Queue
# ---------------------------------------------------------------------------
# All inference requests go through a single-threaded worker to prevent
# concurrent access to the ONNX Runtime sessions and KV cache state.
# Each item on the queue is a tuple: (request_kwargs, result_queue)
# result_queue receives either individual tokens (str) or a sentinel.

_SENTINEL_DONE = "__DONE__"
_SENTINEL_ERROR = "__ERROR__"
_SENTINEL_CANCELLED = "__CANCELLED__"

_inference_queue: queue.Queue = queue.Queue(maxsize=8)
_worker_thread: threading.Thread | None = None

# Per-session cancel tracking.  Maps session_id → cancel_event for the
# currently active (or queued) request in that session.  Requests without a
# session_id are never cancelled by other requests (multi-user safe).
_active_sessions: dict[str, threading.Event] = {}
_sessions_lock = threading.Lock()


def _inference_worker():
    """Background thread that processes inference requests one at a time."""
    global engine
    while True:
        item = _inference_queue.get()
        if item is None:  # shutdown signal
            break
        kwargs, result_q, cancel_event, session_id = item
        # Register this session so future requests can cancel it.
        if session_id:
            with _sessions_lock:
                _active_sessions[session_id] = cancel_event
        try:
            if cancel_event.is_set():
                result_q.put(_SENTINEL_CANCELLED)
                continue
            kwargs["cancel_event"] = cancel_event
            for token in engine.generate_stream(**kwargs):
                if cancel_event.is_set():
                    break
                result_q.put(token)
            if cancel_event.is_set():
                result_q.put(_SENTINEL_CANCELLED)
            else:
                result_q.put(_SENTINEL_DONE)
        except CancelledError:
            result_q.put(_SENTINEL_CANCELLED)
        except GPUMemoryError as e:
            result_q.put((_SENTINEL_ERROR, str(e)))
        except Exception as e:
            traceback.print_exc()
            result_q.put((_SENTINEL_ERROR, f"Inference error: {e}"))
        finally:
            if session_id:
                with _sessions_lock:
                    # Only remove if it's still *our* event (not a newer request).
                    if _active_sessions.get(session_id) is cancel_event:
                        _active_sessions.pop(session_id, None)


def _cancel_session(session_id: str | None):
    """Cancel the active + queued requests for *session_id* only.

    If session_id is None the call is a no-op — anonymous requests are
    never preemptively cancelled (multi-user safe).
    """
    if not session_id:
        return

    # 1. Cancel the currently-running request for this session.
    with _sessions_lock:
        cancel_event = _active_sessions.get(session_id)
        if cancel_event is not None:
            cancel_event.set()

    # 2. Drain queued items for *this session only*; keep everything else.
    kept: list = []
    while True:
        try:
            item = _inference_queue.get_nowait()
            if item is None:
                kept.append(item)  # keep shutdown sentinel
                break
            _, result_q, cancel_ev, sid = item
            if sid == session_id:
                cancel_ev.set()
                result_q.put(_SENTINEL_CANCELLED)
            else:
                kept.append(item)
        except queue.Empty:
            break
    for item in kept:
        _inference_queue.put(item)


async def _enqueue_inference(kwargs: dict, session_id: str | None = None):
    """
    Put a request on the worker queue and yield tokens as they arrive.

    If *session_id* is provided, any previous request **from the same
    session** is cancelled first.  Requests from other sessions (or
    anonymous requests with session_id=None) are left untouched.
    """
    _cancel_session(session_id)

    result_q: queue.Queue = queue.Queue()
    cancel_event = threading.Event()
    loop = asyncio.get_running_loop()

    try:
        _inference_queue.put_nowait((kwargs, result_q, cancel_event, session_id))
    except queue.Full:
        raise RuntimeError(
            "Server is busy — too many concurrent requests. Please try again shortly."
        )

    while True:
        item = await loop.run_in_executor(None, result_q.get)
        if item == _SENTINEL_DONE:
            return
        if item == _SENTINEL_CANCELLED:
            return
        if isinstance(item, tuple) and len(item) == 2 and item[0] == _SENTINEL_ERROR:
            error_msg = item[1]
            if "GPU out of memory" in error_msg:
                raise GPUMemoryError(error_msg)
            raise RuntimeError(error_msg)
        yield item


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup and keep it warm."""
    global engine, _worker_thread
    print(f"[Server] Loading Gemma 4 (Vision Enabled: {ENABLE_VISION}, Audio Enabled: {ENABLE_AUDIO})...")
    model_path = os.path.join(os.path.dirname(__file__), "models", "gemma-4-E2B-it-ONNX")
    if not os.path.exists(model_path):
        model_path = "./models/gemma-4-E2B-it-ONNX"
    engine = GemmaEngine(
        model_path, 
        enable_vision=ENABLE_VISION, 
        enable_audio=ENABLE_AUDIO
    )
    
    # NOTE: Warmup inference removed. With memory arenas disabled the
    # warmup would allocate-then-free, providing no lasting benefit.
    # The first real request pays a small one-time graph-optimization cost.
        
    print("[Server] Starting inference worker thread...")
    _worker_thread = threading.Thread(target=_inference_worker, daemon=True, name="inference-worker")
    _worker_thread.start()

    print("[Server] Model warm, optimized, and ready!")
    yield

    # Shutdown: signal worker to stop
    print("[Server] Shutting down inference worker...")
    _inference_queue.put(None)
    _worker_thread.join(timeout=5)
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
# Request Schemas
# ---------------------------------------------------------------------------
class ToolFunction(BaseModel):
    name: str
    description: str = ""
    parameters: Dict[str, Any] = {}


class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction


class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    model: str = "gemma-4-e2b"
    stream: bool = True
    max_tokens: int = 1024
    temperature: float = 1.0
    visual_token_budget: int = 280
    video_fps: float = 1.0
    enable_thinking: bool = False
    tools: Optional[List[Tool]] = None
    max_context_tokens: int = 8192
    session_id: Optional[str] = None


class OpenAIChatRequest(BaseModel):
    model: str = "gemma-4-e2b"
    messages: List[Dict[str, Any]]
    stream: bool = False
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 1.0
    tools: Optional[List[Tool]] = None
    max_context_tokens: int = 8192
    session_id: Optional[str] = None


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


def _inject_tools(messages: List[Dict[str, Any]], tools: List[Tool]) -> List[Dict[str, Any]]:
    """Serialize tool definitions into a system prompt and inject into messages."""
    tool_defs = [
        {"name": t.function.name, "description": t.function.description, "parameters": t.function.parameters}
        for t in tools
    ]
    tool_prompt = (
        "You have access to the following tools:\n\n"
        + json.dumps(tool_defs, indent=2)
        + "\n\nWhen you want to call a tool, respond ONLY with a JSON object "
        "in this exact format (no other text before or after):\n"
        '{"tool_call": {"name": "<tool_name>", "arguments": {<args_object>}}}\n\n'
        "When you do NOT need a tool, respond normally in plain text."
    )
    injected = list(messages)
    if injected and injected[0].get("role") == "system":
        existing = injected[0]
        if isinstance(existing.get("content"), str):
            injected[0] = {"role": "system", "content": existing["content"] + "\n\n" + tool_prompt}
        else:
            injected[0] = {
                "role": "system",
                "content": list(existing["content"]) + [{"type": "text", "text": "\n\n" + tool_prompt}],
            }
    else:
        injected.insert(0, {"role": "system", "content": tool_prompt})
    return injected


def _try_parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Try to extract a tool_call dict from model output. Returns None if not a tool call."""
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "tool_call" in data:
            return data["tool_call"]
    except (json.JSONDecodeError, ValueError):
        pass
    # Brace-matching scan for any JSON object containing "tool_call"
    for m in re.finditer(r'\{', text):
        depth, start = 0, m.start()
        for i, ch in enumerate(text[start:], start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(text[start:i + 1])
                        if isinstance(data, dict) and "tool_call" in data:
                            return data["tool_call"]
                    except (json.JSONDecodeError, ValueError):
                        pass
                    break
    return None


def _openai_messages_to_internal(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI-format messages (image_url items) to internal format (image items)."""
    result = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            new_content = []
            for item in content:
                if item.get("type") == "image_url":
                    url = (item.get("image_url") or {}).get("url", "")
                    new_content.append({"type": "image", "image": url})
                else:
                    new_content.append(item)
            result.append({"role": msg["role"], "content": new_content})
        else:
            result.append(msg)
    return result


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
                    if m_type == "video":
                        raise ValueError(
                            "Base64-encoded video is not supported. "
                            "Upload via POST /api/upload and reference the returned path."
                        )
                    try:
                        path = _save_base64_to_file(source)
                        temp_files.append(path)
                        new_content.append({"type": m_type, m_type: path})
                        print(f"[Server] Saved {m_type} → {path}")
                    except ValueError:
                        raise
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
    try:
        processed_messages, temp_files = preprocess_messages(
            request.messages, request.video_fps
        )
    except ValueError as ve:
        return {"model": request.model, "message": {"role": "assistant", "content": f"[Server Error: {ve}]"}, "done": True}

    # Inject tool definitions if provided
    final_messages = processed_messages
    if request.tools:
        final_messages = _inject_tools(processed_messages, request.tools)
        print(f"[Server] Injected {len(request.tools)} tool(s) into system prompt")

    print(f"[Server] Chat request: stream={request.stream}, "
          f"thinking={request.enable_thinking}, files={len(temp_files)}")

    # -----------------------------------------------------------------------
    # Streaming response
    # -----------------------------------------------------------------------
    if request.stream:
        async def event_generator():
            token_count = 0
            t0 = time.perf_counter()
            full_text = ""
            try:
                gen_kwargs = dict(
                    messages=final_messages,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    visual_token_budget=request.visual_token_budget,
                    enable_thinking=request.enable_thinking,
                    max_context_tokens=request.max_context_tokens,
                )
                async for token in _enqueue_inference(gen_kwargs, request.session_id):
                    token_count += 1
                    full_text += token
                    chunk = {
                        "model": request.model,
                        "message": {"role": "assistant", "content": token},
                        "done": False,
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                # Check for tool call in full output
                tool_call = _try_parse_tool_call(full_text) if request.tools else None
                elapsed_ns = int((time.perf_counter() - t0) * 1e9)
                done_data = {
                    "model": request.model,
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "eval_count": token_count,
                    "eval_duration": elapsed_ns,
                }
                if tool_call:
                    done_data["message"]["tool_calls"] = [{"function": tool_call}]
                    done_data["finish_reason"] = "tool_calls"
                else:
                    done_data["finish_reason"] = "stop"
                yield f"data: {json.dumps(done_data)}\n\n"
                yield "data: [DONE]\n\n"

            except GPUMemoryError as e:
                err = {"model": request.model,
                       "message": {"role": "assistant",
                                   "content": f"\n\n[GPU Memory Error: {e}]"},
                       "done": True, "finish_reason": "error"}
                yield f"data: {json.dumps(err)}\n\n"
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
        token_count = 0
        t0 = time.perf_counter()
        gen_kwargs = dict(
            messages=final_messages,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            enable_thinking=request.enable_thinking,
            max_context_tokens=request.max_context_tokens,
        )
        async for token in _enqueue_inference(gen_kwargs, request.session_id):
            token_count += 1
            full_text += token

        elapsed_ns = int((time.perf_counter() - t0) * 1e9)
        tool_call = _try_parse_tool_call(full_text) if request.tools else None
        resp = {
            "model": request.model,
            "message": {"role": "assistant", "content": full_text},
            "done": True,
            "eval_count": token_count,
            "eval_duration": elapsed_ns,
        }
        if tool_call:
            resp["message"]["tool_calls"] = [{"function": tool_call}]
            resp["finish_reason"] = "tool_calls"
        else:
            resp["finish_reason"] = "stop"
        return resp
    except GPUMemoryError as e:
        return {"model": request.model,
                "message": {"role": "assistant", "content": f"[GPU Memory Error: {e}]"},
                "done": True, "finish_reason": "error"}
    except Exception as e:
        traceback.print_exc()
        return {"model": request.model,
                "message": {"role": "assistant", "content": f"[Error: {e}]"},
                "done": True}
    finally:
        _cleanup_files(temp_files)


# ---------------------------------------------------------------------------
# Ollama Compatibility Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/tags")
async def api_tags():
    """Ollama-compatible endpoint listing available models."""
    return {
        "models": [
            {
                "name": "gemma-4-e2b",
                "modified_at": "2025-06-01T00:00:00Z",
                "size": 4_000_000_000,
                "digest": "gemma-4-e2b-onnx-q4",
                "details": {
                    "format": "onnx",
                    "family": "gemma4",
                    "parameter_size": "4B",
                    "quantization_level": "q4",
                },
            }
        ]
    }


@app.post("/api/show")
async def api_show(request: Dict[str, Any]):
    """Ollama-compatible endpoint returning model metadata."""
    model_name = request.get("name", "gemma-4-e2b")
    return {
        "modelfile": f"# {model_name}",
        "parameters": "num_ctx 8192",
        "template": "<start_of_turn>user\n{{.Prompt}}<end_of_turn>\n<start_of_turn>model",
        "details": {
            "format": "onnx",
            "family": "gemma4",
            "parameter_size": "4B",
            "quantization_level": "q4",
        },
        "model_info": {
            "general.architecture": "gemma4",
            "general.parameter_count": 4_000_000_000,
            "gemma.context_length": 8192,
        },
    }


# ---------------------------------------------------------------------------
# OpenAI-Compatible Endpoints
# ---------------------------------------------------------------------------
@app.get("/v1/models")
async def v1_models():
    """OpenAI-compatible models list endpoint."""
    return {
        "object": "list",
        "data": [
            {
                "id": "gemma-4-e2b",
                "object": "model",
                "created": 1717200000,
                "owned_by": "google",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def v1_chat_completions(request: OpenAIChatRequest):
    """
    OpenAI-compatible chat completion endpoint.
    Converts to internal format and forwards to the engine.
    """
    global engine
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": {"message": "Model not loaded yet", "type": "server_error"}},
        )

    # Convert messages from OpenAI format (image_url) → internal (image)
    internal_msgs = _openai_messages_to_internal(request.messages)

    # Preprocess (base64 save, video expand)
    try:
        processed, temp_files = preprocess_messages(internal_msgs)
    except ValueError as ve:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": str(ve), "type": "invalid_request_error"}},
        )

    final_messages = processed
    if request.tools:
        final_messages = _inject_tools(processed, request.tools)

    # -----------------------------------------------------------------------
    # Streaming response (SSE in OpenAI format)
    # -----------------------------------------------------------------------
    if request.stream:
        async def openai_stream():
            chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())
            full_text = ""
            try:
                gen_kwargs = dict(
                    messages=final_messages,
                    max_new_tokens=request.max_tokens or 1024,
                    temperature=request.temperature or 1.0,
                    max_context_tokens=request.max_context_tokens,
                )
                async for token in _enqueue_inference(gen_kwargs, request.session_id):
                    full_text += token
                    chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                # Final chunk
                tool_call = _try_parse_tool_call(full_text) if request.tools else None
                finish = {"index": 0, "delta": {}, "finish_reason": "tool_calls" if tool_call else "stop"}
                if tool_call:
                    finish["delta"]["tool_calls"] = [{"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function", "function": tool_call}]
                final_chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [finish],
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            except GPUMemoryError as e:
                err_chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"content": f"\n\n[GPU Memory Error: {e}]"}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(err_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as e:
                traceback.print_exc()
                err_chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"content": f"\n\n[Error: {e}]"}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(err_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                _cleanup_files(temp_files)

        return StreamingResponse(openai_stream(), media_type="text/event-stream")

    # -----------------------------------------------------------------------
    # Non-streaming response
    # -----------------------------------------------------------------------
    try:
        t0 = time.perf_counter()
        full_text = ""
        token_count = 0
        gen_kwargs = dict(
            messages=final_messages,
            max_new_tokens=request.max_tokens or 1024,
            temperature=request.temperature or 1.0,
            max_context_tokens=request.max_context_tokens,
        )
        async for token in _enqueue_inference(gen_kwargs, request.session_id):
            full_text += token
            token_count += 1

        tool_call = _try_parse_tool_call(full_text) if request.tools else None
        message: Dict[str, Any] = {"role": "assistant", "content": full_text}
        finish_reason = "stop"
        if tool_call:
            message["tool_calls"] = [{"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function", "function": tool_call}]
            finish_reason = "tool_calls"

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": 0,  # Not tracked
                "completion_tokens": token_count,
                "total_tokens": token_count,
            },
        }
    except GPUMemoryError as e:
        return JSONResponse(
            status_code=507,
            content={"error": {"message": str(e), "type": "insufficient_resources"}},
        )
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "server_error"}},
        )
    finally:
        _cleanup_files(temp_files)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
