/**
 * Gemma 4 Chat UI — Frontend Logic
 * Connects to the local Ollama-compatible server at /api/chat
 * Supports: streaming, thinking mode, image/audio upload, multi-turn history
 */

// If we're NOT being served by the Gemma server itself (port 11434),
// explicitly point all API calls at it (covers file://, Live Server, etc.)
const GEMMA_PORT = 11435;
const API_BASE =
  window.location.port == GEMMA_PORT
    ? ""
    : `http://localhost:${GEMMA_PORT}`;

// ── State ───────────────────────────────────────────────────────────────────

/** @type {Array<{role: string, content: string}>} */
let history = [];
let pendingMedia = null; // { type: "image"|"audio", dataUrl: string, mime: string }
let isGenerating = false;

// ── DOM Refs ────────────────────────────────────────────────────────────────

const messagesEl   = document.getElementById("messages");
const promptInput  = document.getElementById("promptInput");
const sendBtn      = document.getElementById("sendBtn");
const sendIcon     = document.getElementById("sendIcon");
const sendSpinner  = document.getElementById("sendSpinner");
const mediaFile    = document.getElementById("mediaFile");
const mediaStrip   = document.getElementById("mediaStrip");
const mediaThumb   = document.getElementById("mediaThumb");
const removeMedia  = document.getElementById("removeMedia");
const welcomeCard  = document.getElementById("welcomeCard");
const clearBtn     = document.getElementById("clearBtn");
const menuBtn      = document.getElementById("menuBtn");
const sidebar      = document.getElementById("sidebar");

// Status
const statusDot    = document.getElementById("statusDot");
const statusLabel  = document.getElementById("statusLabel");
const statusDetail = document.getElementById("statusDetail");
const progressWrap = document.getElementById("progressWrap");
const progressFill = document.getElementById("progressFill");
const topDot       = document.getElementById("topDot");
const topStatus    = document.getElementById("topStatus");

// Sliders
const maxTokensEl   = document.getElementById("maxTokens");
const temperatureEl = document.getElementById("temperature");
const topPEl        = document.getElementById("topP");
const maxTokensVal  = document.getElementById("maxTokensVal");
const temperatureVal= document.getElementById("temperatureVal");
const topPVal       = document.getElementById("topPVal");

// ── Status Polling ──────────────────────────────────────────────────────────

/** Poll /api/status every 3 seconds until the model is ready */
async function pollStatus() {
  try {
    const res = await fetch(`${API_BASE}/api/status`);
    if (!res.ok) throw new Error("Server unreachable");
    const data = await res.json();
    const m = data.model;

    if (m.loaded) {
      setStatus("ready", "Model Ready", m.modelId.split("/").pop());
      return; // stop polling
    } else if (m.loading) {
      const pct = m.progress ?? 0;
      setStatus("loading", "Loading Model…", `${pct}% — This may take a few minutes on first run`);
      progressWrap.style.display = "block";
      progressFill.style.width = `${pct}%`;
    } else {
      setStatus("loading", "Initializing…", "Starting model load...");
    }
  } catch {
    setStatus("error", "Server Offline", "Make sure `npm start` is running");
  }
  setTimeout(pollStatus, 3000);
}

/** @param {"ready"|"loading"|"error"} type */
function setStatus(type, label, detail) {
  statusDot.className = `status-dot ${type}`;
  statusLabel.textContent = label;
  statusDetail.textContent = detail;
  topDot.className = `indicator-dot ${type}`;
  topStatus.textContent = label;
  if (type !== "loading") progressWrap.style.display = "none";
}

// ── Slider bindings ─────────────────────────────────────────────────────────

maxTokensEl.addEventListener("input", () => {
  maxTokensVal.textContent = maxTokensEl.value;
});
temperatureEl.addEventListener("input", () => {
  temperatureVal.textContent = parseFloat(temperatureEl.value).toFixed(2);
});
topPEl.addEventListener("input", () => {
  topPVal.textContent = parseFloat(topPEl.value).toFixed(2);
});

// ── Media Upload ─────────────────────────────────────────────────────────────

mediaFile.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (ev) => {
    pendingMedia = {
      type: file.type.startsWith("image") ? "image" : "audio",
      dataUrl: ev.target.result,
      mime: file.type,
      name: file.name,
    };
    const icon = pendingMedia.type === "image" ? "🖼️" : "🎵";
    mediaThumb.textContent = `${icon} ${file.name} (${(file.size / 1024).toFixed(0)} KB)`;
    mediaStrip.style.display = "flex";
  };
  reader.readAsDataURL(file);
  mediaFile.value = ""; // reset so same file can be re-uploaded
});

removeMedia.addEventListener("click", () => {
  pendingMedia = null;
  mediaStrip.style.display = "none";
});

// ── Sidebar / Menu ──────────────────────────────────────────────────────────

menuBtn.addEventListener("click", () => {
  sidebar.classList.toggle("open");
});

// ── Welcome Chips ────────────────────────────────────────────────────────────

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    promptInput.value = chip.dataset.prompt;
    promptInput.dispatchEvent(new Event("input"));
    promptInput.focus();
  });
});

// ── Clear Chat ───────────────────────────────────────────────────────────────

clearBtn.addEventListener("click", () => {
  history = [];
  // Remove all messages except welcome card
  Array.from(messagesEl.children).forEach((el) => {
    if (el.id !== "welcomeCard") el.remove();
  });
  welcomeCard.style.display = "";
});

// ── Auto-resize textarea ─────────────────────────────────────────────────────

promptInput.addEventListener("input", () => {
  promptInput.style.height = "auto";
  promptInput.style.height = Math.min(promptInput.scrollHeight, 160) + "px";
});

// ── Send on Enter (Shift+Enter = newline) ────────────────────────────────────

promptInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!isGenerating) handleSend();
  }
});

sendBtn.addEventListener("click", () => {
  if (!isGenerating) handleSend();
});

// ── Core Send Logic ──────────────────────────────────────────────────────────

async function handleSend() {
  const prompt = promptInput.value.trim();
  if (!prompt && !pendingMedia) return;

  const enableThinking = document.getElementById("enableThinking").checked;
  const enableStream   = document.getElementById("enableStream").checked;
  const maxTokens      = parseInt(maxTokensEl.value);
  const temperature    = parseFloat(temperatureEl.value);
  const topP           = parseFloat(topPEl.value);

  // Capture media and clear strip
  const media = pendingMedia;
  pendingMedia = null;
  mediaStrip.style.display = "none";

  // Clear input
  promptInput.value = "";
  promptInput.style.height = "auto";

  // Hide welcome card on first message
  if (welcomeCard) welcomeCard.style.display = "none";

  // Render user message
  appendUserMessage(prompt, media);

  // Build message content for API
  let content;
  if (media) {
    const userContent = [];
    if (media.type === "image") {
      userContent.push({ type: "image", image: media.dataUrl });
    } else if (media.type === "audio") {
      userContent.push({ type: "audio", audio: media.dataUrl });
    }
    if (prompt) userContent.push({ type: "text", text: prompt });
    content = userContent;
  } else {
    content = prompt;
  }

  history.push({ role: "user", content: content });

  // Show generation
  setGenerating(true);

  try {
    if (enableStream) {
      await streamResponse({ enableThinking, maxTokens, temperature, topP });
    } else {
      await fetchResponse({ enableThinking, maxTokens, temperature, topP });
    }
  } catch (err) {
    appendErrorMessage(err.message);
  } finally {
    setGenerating(false);
  }
}

// ── Streaming Response ────────────────────────────────────────────────────────

async function streamResponse({ enableThinking, maxTokens, temperature, topP }) {
  const payload = {
    model: "gemma4:e2b",
    messages: history,
    stream: true,
    think: enableThinking,
    options: {
      num_predict: maxTokens,
      temperature,
      top_p: topP,
      top_k: 64,
    },
  };

  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
    throw new Error(err.error ?? `Server error ${res.status}`);
  }

  // Create AI message bubble with streaming cursor
  const { bubble, cursor, removeCursor } = createAIBubble();

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let fullContent = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value, { stream: true });
    const lines = chunk.split("\n").filter((l) => l.trim());

    for (let line of lines) {
      if (line.startsWith("data: ")) line = line.slice(6);
      if (line === "[DONE]") break;
      try {
        const data = JSON.parse(line);
        if (data.error) throw new Error(data.error);

        const token = data.message?.content ?? "";
        if (token) {
          fullContent += token;
          bubble.innerHTML = marked.parse(fullContent);
          bubble.appendChild(cursor);
          scrollToBottom();
        }

        if (data.done) {
          removeCursor();
          // Save to history (plain text for context)
          history.push({ role: "assistant", content: [{ type: "text", text: fullContent }] });
          break;
        }
      } catch {
        // skip malformed JSON lines
      }
    }
  }
}

// ── Non-streaming Response ────────────────────────────────────────────────────

async function fetchResponse({ enableThinking, maxTokens, temperature, topP }) {
  const payload = {
    model: "gemma4:e2b",
    messages: history,
    stream: false,
    think: enableThinking,
    options: { num_predict: maxTokens, temperature, top_p: topP, top_k: 64 },
  };

  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
    throw new Error(err.error ?? `Server error ${res.status}`);
  }

  const data = await res.json();
  const content = data.message?.content ?? "";
  const thinking = data.message?.thinking ?? "";

  appendAIMessage(content, thinking);
  history.push({ role: "assistant", content: [{ type: "text", text: content }] });
}

// ── Message Rendering ────────────────────────────────────────────────────────

function appendUserMessage(text, media) {
  const el = document.createElement("div");
  el.className = "message";
  let mediaHTML = "";
  if (media?.type === "image") {
    mediaHTML = `<img class="msg-image" src="${escapeHTML(media.dataUrl)}" alt="uploaded image" />`;
  }
  el.innerHTML = `
    <div class="message-header">
      <div class="avatar avatar-user">U</div>
      <span>You</span>
    </div>
    <div class="bubble bubble-user">
      ${mediaHTML}
      ${text ? `<span>${escapeHTML(text)}</span>` : ""}
    </div>
  `;
  messagesEl.appendChild(el);
  scrollToBottom();
}

/**
 * Creates a streaming AI message bubble. Returns refs for live-updating.
 * @returns {{ bubble: HTMLElement, cursor: HTMLElement, removeCursor: Function }}
 */
function createAIBubble() {
  const el = document.createElement("div");
  el.className = "message";

  const bubble = document.createElement("div");
  bubble.className = "bubble bubble-ai";

  const cursor = document.createElement("span");
  cursor.className = "cursor";

  bubble.appendChild(cursor);
  el.innerHTML = `
    <div class="message-header">
      <div class="avatar avatar-ai">G4</div>
      <span>Gemma 4</span>
    </div>
  `;
  el.appendChild(bubble);
  messagesEl.appendChild(el);
  scrollToBottom();

  return {
    bubble,
    cursor,
    removeCursor: () => cursor.remove(),
  };
}

function appendAIMessage(content, thinking = "") {
  const el = document.createElement("div");
  el.className = "message";

  let thinkHTML = "";
  if (thinking) {
    thinkHTML = `
      <div class="think-block">
        <div class="think-label">⚡ Internal Reasoning</div>
        <div class="think-content">${escapeHTML(thinking)}</div>
      </div>
    `;
  }

  el.innerHTML = `
    <div class="message-header">
      <div class="avatar avatar-ai">G4</div>
      <span>Gemma 4</span>
    </div>
    ${thinkHTML}
    <div class="bubble bubble-ai">${marked.parse(content)}</div>
  `;
  messagesEl.appendChild(el);
  scrollToBottom();
}

function appendErrorMessage(msg) {
  const el = document.createElement("div");
  el.className = "message";
  el.innerHTML = `
    <div class="message-header">
      <div class="avatar" style="background:#ef4444;color:white">!</div>
      <span style="color:#f87171">Error</span>
    </div>
    <div class="bubble" style="border:1px solid rgba(248,113,113,0.3);background:rgba(248,113,113,0.07);color:#fca5a5">
      ${escapeHTML(msg)}
    </div>
  `;
  messagesEl.appendChild(el);
  scrollToBottom();
}

// ── Utils ────────────────────────────────────────────────────────────────────

function setGenerating(val) {
  isGenerating = val;
  sendBtn.disabled = val;
  sendIcon.style.display = val ? "none" : "block";
  sendSpinner.style.display = val ? "block" : "none";
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function escapeHTML(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ── Init ─────────────────────────────────────────────────────────────────────

pollStatus();
