
# Gemma 4 E2B Multimodal Server — API Documentation

## Server Startup

```powershell
# Full multimodal
uv run python -m uvicorn app:app --host 0.0.0.0 --port 8000

# Vision + text only (saves VRAM)
$env:GEMMA_ENABLE_AUDIO="false"; uv run python -m uvicorn app:app --host 0.0.0.0 --port 8000

# Audio + text only (saves VRAM)
$env:GEMMA_ENABLE_VISION="false"; uv run python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### Per-Modality Device Routing (`engine_config.json`)

Place an `engine_config.json` in the project root to route individual ONNX sessions to CPU or GPU.
This is useful when GPU VRAM is limited — offload a modality to CPU to free memory for the others.

```json
{
  "text": "cpu",
  "image": "gpu",
  "audio": "cpu"
}
```

| Key | Values | Controls |
|---|---|---|
| `text` | `"cpu"` / `"gpu"` | `embed_tokens` + `decoder` sessions |
| `image` | `"cpu"` / `"gpu"` | `vision_encoder` session |
| `audio` | `"cpu"` / `"gpu"` | `audio_encoder` session |

Missing keys default to `"gpu"` when running with CUDA. No API changes required.

---

## Routes

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI (serves index.html) |
| `POST` | `/api/upload` | Upload file, returns server path |
| `POST` | `/api/chat` | Ollama-style chat inference |
| `GET` | `/api/tags` | Ollama-compatible model list |
| `POST` | `/api/show` | Ollama-compatible model metadata |
| `GET` | `/v1/models` | OpenAI-compatible model list |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat |

---

## `POST /api/upload`

Upload a file (image, audio, video) to reference in chat messages.

**Request:** `multipart/form-data` with field `file`

**Response:**
```json
{"path": "uploads/550e8400-e29b.jpg", "size": 204800}
```

---

## `POST /api/chat`

Primary chat endpoint (Ollama-compatible).

### Request Body

```json
{
  "messages": [...],
  "model": "gemma-4-e2b",
  "stream": true,
  "max_tokens": 1024,
  "temperature": 1.0,
  "visual_token_budget": 280,
  "video_fps": 1.0,
  "enable_thinking": false,
  "tools": null,
  "max_context_tokens": 8192,
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `messages` | array | **required** | Conversation turns |
| `model` | string | `"gemma-4-e2b"` | Model name (echoed in response) |
| `stream` | bool | `true` | SSE streaming vs full JSON |
| `max_tokens` | int | `1024` | Max **new** tokens to generate (output cap) |
| `temperature` | float | `1.0` | 0 = greedy, higher = more random |
| `visual_token_budget` | int | `280` | Visual patch tokens (280–2048+) |
| `video_fps` | float | `1.0` | Frames/sec extracted from video |
| `enable_thinking` | bool | `false` | Enable `<|think|>` reasoning mode |
| `tools` | array | `null` | Tool definitions for function calling |
| `max_context_tokens` | int | `8192` | Max **input** context window — truncates from left if exceeded |
| `session_id` | string | `null` | Session identifier for request cancellation (see [Session-Scoped Cancellation](#session-scoped-cancellation)) |

> **`max_tokens` vs `max_context_tokens`** — these are different controls:
>
> - `max_tokens` caps how many **new tokens the model generates** (output length).
> - `max_context_tokens` caps how many **input tokens** are fed to the model. When the conversation history exceeds this, older tokens are dropped from the left. For multimodal inputs, truncation is skipped to preserve token alignment.
>
> Setting `max_tokens` to the context slider value is a common mistake — it causes the model to attempt generating thousands of tokens, which grows the KV cache and triggers GPU OOM.

### Message Format

#### Correct Client Usage

```js
// ✅ Correct — max_tokens caps generation, max_context_tokens caps input window
const SESSION_ID = crypto.randomUUID();  // generate once per page load

fetch("/api/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    messages: chatHistory,
    max_tokens: 1024,                              // output cap (generation length)
    max_context_tokens: parseInt(contextSlider.value), // input cap (context window)
    temperature: 0.7,
    session_id: SESSION_ID,                         // enables cancel-and-replace
  })
});

// ❌ Wrong — sends context slider as max_tokens (causes OOM on long conversations)
// max_tokens: parseInt(contextSlider.value),  // DON'T — this is generation length, not context
```

### Message Format

```json
{
  "role": "user|assistant|system",
  "content": "string" | [...]
}
```

`content` can be a plain string OR an array of typed items:

| `type` | Payload Key | Value |
|---|---|---|
| `text` | `text` | The text string |
| `image` | `image` | Base64 data URI or server path |
| `audio` | `audio` | Base64 data URI or server path |
| `video` | `video` | Server path only (no base64) |

### Streaming Response (SSE)

```
data: {"model": "gemma-4-e2b", "message": {"role": "assistant", "content": "token"}, "done": false}
data: {"model": "gemma-4-e2b", "message": {...}, "done": true, "eval_count": 42, "eval_duration": 1234567890, "finish_reason": "stop"}
data: [DONE]
```

### Non-Streaming Response

```json
{
  "model": "gemma-4-e2b",
  "message": {"role": "assistant", "content": "Full response"},
  "done": true,
  "eval_count": 42,
  "eval_duration": 1234567890,
  "finish_reason": "stop"
}
```

---

## `POST /v1/chat/completions`

OpenAI-compatible endpoint. Works with OpenAI SDKs.

### Request Body

```json
{
  "model": "gemma-4-e2b",
  "messages": [...],
  "stream": false,
  "max_tokens": 1024,
  "temperature": 1.0,
  "tools": null,
  "max_context_tokens": 8192,
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `messages` | array | **required** | Conversation turns |
| `model` | string | `"gemma-4-e2b"` | Model name |
| `stream` | bool | `false` | SSE streaming vs full JSON |
| `max_tokens` | int | `1024` | Max tokens to generate |
| `temperature` | float | `1.0` | Sampling temperature |
| `tools` | array | `null` | Tool definitions for function calling |
| `max_context_tokens` | int | `8192` | Max input context window |
| `session_id` | string | `null` | Session identifier for request cancellation (see [Session-Scoped Cancellation](#session-scoped-cancellation)) |

### OpenAI Message Format

Images use `image_url` type (auto-converted internally):

```json
{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
```

### Streaming Response

```
data: {"id": "chatcmpl-abc123", "object": "chat.completion.chunk", "model": "gemma-4-e2b", "choices": [{"index": 0, "delta": {"content": "token"}, "finish_reason": null}]}
data: [DONE]
```

### Non-Streaming Response

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1715600000,
  "model": "gemma-4-e2b",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 0, "completion_tokens": 42, "total_tokens": 42}
}
```

---

## `GET /api/tags`

Returns available models (Ollama format).

```json
{
  "models": [{
    "name": "gemma-4-e2b",
    "size": 4000000000,
    "details": {"format": "onnx", "family": "gemma4", "parameter_size": "4B", "quantization_level": "q4"}
  }]
}
```

---

## `POST /api/show`

Returns model metadata (Ollama format).

**Request:** `{"name": "gemma-4-e2b"}`

**Response:**
```json
{
  "modelfile": "# gemma-4-e2b",
  "parameters": "num_ctx 8192",
  "template": "<start_of_turn>user\n{{.Prompt}}<end_of_turn>\n<start_of_turn>model",
  "model_info": {"gemma.context_length": 8192}
}
```

---

## `GET /v1/models`

OpenAI-compatible models list.

```json
{
  "object": "list",
  "data": [{"id": "gemma-4-e2b", "object": "model", "owned_by": "google"}]
}
```

---

## Tool Calling

Pass tool definitions in the `tools` field. The model outputs JSON when it wants to call a tool.

### Tool Schema

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          },
          "required": ["location"]
        }
      }
    }
  ]
}
```

### Tool Call Response

When the model calls a tool, `finish_reason` is `"tool_calls"` and the message includes:

```json
{
  "message": {
    "role": "assistant",
    "content": "{\"tool_call\": ...}",
    "tool_calls": [{"function": {"name": "get_weather", "arguments": {"location": "Tokyo"}}}]
  },
  "finish_reason": "tool_calls"
}
```

### Sending Tool Results

Add the result as a follow-up user message:

```json
{"role": "user", "content": "Tool result: {\"temperature\": \"18°C\"}"}
```

---

## Usage Examples

### Text Only

```json
{
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is 2+2?"}
  ]
}
```

### Image + Text

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe this image."},
      {"type": "image", "image": "data:image/jpeg;base64,/9j/4AAQ..."}
    ]
  }],
  "visual_token_budget": 512
}
```

### Audio + Text

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Transcribe this audio."},
      {"type": "audio", "audio": "uploads/speech.wav"}
    ]
  }]
}
```

### Full Multimodal (Audio + Image + Text)

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe both the audio and image."},
      {"type": "audio", "audio": "uploads/speech.wav"},
      {"type": "image", "image": "uploads/photo.jpg"}
    ]
  }],
  "visual_token_budget": 400,
  "enable_thinking": true
}
```

### Video

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Summarize this video."},
      {"type": "video", "video": "uploads/clip.mp4"}
    ]
  }],
  "video_fps": 2.0,
  "visual_token_budget": 1024
}
```

---

## Key Parameters Explained

### `visual_token_budget`
Controls how many visual patch tokens the image encoder produces. Higher values = more detail but more VRAM and latency.

| Value | Use Case |
|---|---|
| `280` (default) | Simple object identification, thumbnails |
| `512` | Charts, diagrams, documents with some text |
| `1024` | Dense text in images, detailed scenes |
| `2048+` | Maximum fidelity (use only if VRAM allows) |

For **video**: multiply by frame count. A 10-second clip at 2 fps = 20 frames × budget. Set conservatively or the KV-cache will OOM.

---

### `max_context_tokens`
Maximum number of input tokens before truncation. Default `8192`.

- When exceeded with **text-only** input: oldest tokens are dropped from the left.
- When exceeded with **multimodal** input: truncation is skipped (would break token alignment). You'll get a warning log instead.

---

### `temperature`
Controls randomness in token sampling.

| Value | Behavior |
|---|---|
| `0` | Greedy decoding (always pick highest probability token) — deterministic output |
| `0.3–0.7` | Low randomness, focused responses |
| `1.0` (default) | Balanced sampling |
| `1.2–2.0` | More creative, varied, potentially incoherent |

---

### `max_tokens`
Maximum number of tokens to generate. Default `1024`. Generation stops earlier if EOS token is reached.

---

### `enable_thinking`
When `true`, injects a `<|think|>` control token into the system prompt. The model then outputs a reasoning block before its final answer:

```
<think>
Let me work through this step by step...
3x + 7 = 22
3x = 15
x = 5
</think>
The answer is x = 5.
```

Your client should parse and optionally hide the `<think>...</think>` block.

---

### `session_id` — Session-Scoped Cancellation

The server processes inference requests one at a time through a single worker thread. The optional `session_id` parameter controls **cancel-and-replace** behavior:

| Scenario | Behavior |
|---|---|
| New request arrives **with `session_id`** that matches an in-flight request | The old request is cancelled mid-generation, the new one takes its place |
| New request arrives **with a different `session_id`** | The new request queues behind the in-flight one — no cancellation |
| New request arrives **without `session_id`** (`null`) | The new request queues normally — never cancels anything |

This makes the server **multi-user safe**: two browser tabs (different session IDs) queue their requests fairly, while rapid-fire messages within the same tab cancel-and-replace for responsiveness.

#### How to use it

**Browser frontend:** Generate a UUID once per page load and send it with every request.

```js
const SESSION_ID = crypto.randomUUID();

// Every chat request includes the same session_id
fetch("/api/chat", {
  body: JSON.stringify({ messages, session_id: SESSION_ID, ... })
});
```

**OpenAI SDK / external clients:** Omit `session_id` entirely — requests will queue FIFO without any cancellation. This is the safe default for multi-user or headless API usage.

```python
# Python (openai SDK) — no session_id, requests queue normally
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
client.chat.completions.create(
    model="gemma-4-e2b",
    messages=[{"role": "user", "content": "Hello"}],
)
```

**Cancellation flow:**
1. Client sends request A with `session_id: "abc"`
2. Server starts generating tokens for A
3. Client sends request B with `session_id: "abc"`
4. Server cancels A (stops generation, frees resources), starts B
5. Client receives the SSE stream for B only

For the built-in web UI, the frontend also uses `AbortController` to close the HTTP connection client-side when a new message is sent, ensuring no stale tokens leak to the UI.

---

### `video_fps`
Frames per second extracted from video files. Default `1.0`.

| Value | Effect |
|---|---|
| `0.5` | 1 frame every 2 seconds (sparse, fast) |
| `1.0` | 1 frame per second |
| `2.0` | 2 frames per second (more detail, 2× tokens) |

Higher fps = more frames = more `visual_token_budget` consumption. For a 30-second video at 2 fps with budget 280: `60 frames × 280 = 16,800 visual tokens`.

---

### `stream`
- `true` (default for `/api/chat`): Server-Sent Events, tokens delivered incrementally.
- `false` (default for `/v1/chat/completions`): Single JSON response after full generation.

---

### `tools`
Array of tool/function definitions. When provided:
1. Tool schemas are serialized into the system prompt
2. Model may output `{"tool_call": {"name": "...", "arguments": {...}}}`
3. Response includes `"finish_reason": "tool_calls"` and parsed `tool_calls` array
4. Your client executes the tool and sends results back as a follow-up message

---

### `model`
Model identifier string. Currently only `"gemma-4-e2b"` is supported. This value is echoed in responses for client compatibility.