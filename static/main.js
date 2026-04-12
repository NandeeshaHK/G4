/* =========================================================================
   Gemma 4 Multimodal Hub – Frontend Logic
   
   Key design: Files are uploaded to /api/upload FIRST, then only small
   server-side paths are sent in chat messages. This prevents the browser
   from freezing when serializing large base64 payloads.
   ========================================================================= */

const API_BASE = 'http://127.0.0.1:8000';
const budgetOptions = [70, 140, 280, 560, 1120];
let currentFiles = [];   // { type, data (for preview), serverPath (after upload) }
let chatHistory = [];

// UI Elements
const visualSlider = document.getElementById('visualBudget');
const visualValue = document.getElementById('visualBudgetValue');
const contextSlider = document.getElementById('contextLength');
const contextValue = document.getElementById('contextLengthValue');
const fpsSlider = document.getElementById('videoFps');
const fpsValue = document.getElementById('videoFpsValue');
const tempSlider = document.getElementById('temperature');
const tempValue = document.getElementById('temperatureValue');

// Update value displays
visualSlider.oninput = () => visualValue.innerText = budgetOptions[visualSlider.value] + " tokens";
contextSlider.oninput = () => contextValue.innerText = contextSlider.value + " tokens";
fpsSlider.oninput = () => fpsValue.innerText = parseFloat(fpsSlider.value).toFixed(1) + " fps";
tempSlider.oninput = () => tempValue.innerText = parseFloat(tempSlider.value).toFixed(1);

/* ====================== File Handling ====================== */

async function handleFiles(files) {
    const preview = document.getElementById('previewContainer');
    preview.innerHTML = '';
    currentFiles = [];

    for (const file of files) {
        // Create preview
        const div = document.createElement('div');
        div.style.position = 'relative';
        const mediaType = file.type.split('/')[0]; // image, audio, video

        if (mediaType === 'image') {
            const img = document.createElement('img');
            img.src = URL.createObjectURL(file);
            img.className = 'file-preview';
            div.appendChild(img);
        } else if (mediaType === 'audio') {
            div.innerHTML = `<div class="file-preview" style="display:flex;align-items:center;justify-content:center;background:rgba(255,255,255,0.1)">🎤</div>`;
        } else if (mediaType === 'video') {
            div.innerHTML = `<div class="file-preview" style="display:flex;align-items:center;justify-content:center;background:rgba(255,255,255,0.1)">🎬</div>`;
        }
        preview.appendChild(div);

        // Store the raw File object (NOT base64) — we'll upload it later
        currentFiles.push({
            type: mediaType,
            file: file,
            previewUrl: mediaType === 'image' ? URL.createObjectURL(file) : null,
            serverPath: null, // will be set after upload
        });
    }
}

/**
 * Upload all pending files to the server and get back server-side paths.
 * Returns true on success.
 */
async function uploadPendingFiles() {
    for (const f of currentFiles) {
        if (f.serverPath) continue; // already uploaded

        const formData = new FormData();
        formData.append('file', f.file);

        try {
            const resp = await fetch(`${API_BASE}/api/upload`, {
                method: 'POST',
                body: formData,
            });
            if (!resp.ok) throw new Error(`Upload failed: ${resp.status}`);
            const data = await resp.json();
            f.serverPath = data.path;
            console.log(`[Upload] ${f.file.name} → ${data.path} (${data.size} bytes)`);
        } catch (e) {
            console.error('[Upload Error]', e);
            return false;
        }
    }
    return true;
}

/* ====================== Send Message ====================== */

async function sendMessage() {
    const input = document.getElementById('userInput');
    const text = input.value.trim();
    if (!text && currentFiles.length === 0) return;

    // Show user message in UI immediately
    appendMessage('user', text, currentFiles);
    input.value = '';

    // Upload files first (non-blocking for the user — they see the preview)
    const filesToSend = [...currentFiles];
    currentFiles = [];
    document.getElementById('previewContainer').innerHTML = '';

    // Create AI placeholder
    const aiMsgDiv = appendMessage('ai', '', []);
    const statusSpan = aiMsgDiv.querySelector('span');
    statusSpan.innerText = '⏳ Uploading files...';

    if (filesToSend.length > 0) {
        // Upload files to server
        for (const f of filesToSend) {
            if (f.serverPath) continue;
            const formData = new FormData();
            formData.append('file', f.file);
            try {
                const resp = await fetch(`${API_BASE}/api/upload`, {
                    method: 'POST',
                    body: formData,
                });
                if (!resp.ok) throw new Error(`Upload failed: ${resp.status}`);
                const data = await resp.json();
                f.serverPath = data.path;
            } catch (e) {
                statusSpan.innerText = `❌ File upload failed: ${e.message}`;
                return;
            }
        }
    }

    statusSpan.innerText = '⏳ Generating...';

    // Build content array with SERVER PATHS (not base64)
    // Place media BEFORE text for optimal performance (per README)
    const content = [];
    filesToSend.forEach(f => {
        if (f.serverPath) {
            content.push({ type: f.type, [f.type]: f.serverPath });
        }
    });
    if (text) content.push({ type: 'text', text: text });

    const userMsg = { role: 'user', content: content };
    chatHistory.push(userMsg);

    let fullContent = "";

    try {
        const response = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                messages: chatHistory,
                max_tokens: parseInt(contextSlider.value),
                temperature: parseFloat(tempSlider.value),
                visual_token_budget: budgetOptions[visualSlider.value],
                video_fps: parseFloat(fpsSlider.value),
                enable_thinking: document.getElementById('thinkingMode').checked
            })
        });

        if (!response.ok) {
            statusSpan.innerText = `❌ Server error: ${response.status}`;
            return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed || !trimmed.startsWith('data: ')) continue;

                const dataStr = trimmed.substring(6);
                if (dataStr === '[DONE]') continue;

                try {
                    const data = JSON.parse(dataStr);
                    const token = data.message.content;
                    if (token) {
                        fullContent += token;
                        renderContent(aiMsgDiv, fullContent);
                    }
                } catch (e) { /* partial JSON */ }
            }
        }

        if (fullContent) {
            // Strip thinking from history (per README: no thinking in multi-turn)
            let historyContent = fullContent;
            const thinkMatch = historyContent.match(/<\|channel>thought\n[\s\S]*?<channel\|>/);
            if (thinkMatch) {
                historyContent = historyContent.replace(thinkMatch[0], '').trim();
            }
            chatHistory.push({ role: 'assistant', content: historyContent });
        } else {
            statusSpan.innerText = '⚠️ No response received.';
        }

    } catch (err) {
        console.error('[Chat Error]', err);
        statusSpan.innerText = `❌ Connection error: ${err.message}`;
    }
}

/* ====================== Message Rendering ====================== */

function appendMessage(role, text, files) {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = `${role}-message message`;

    if (files && files.length > 0) {
        files.forEach(f => {
            if (f.type === 'image' && f.previewUrl) {
                const img = document.createElement('img');
                img.src = f.previewUrl;
                img.style.maxWidth = '200px';
                img.style.borderRadius = '8px';
                img.style.display = 'block';
                img.style.marginBottom = '8px';
                div.appendChild(img);
            } else if (f.type === 'audio') {
                const badge = document.createElement('div');
                badge.style.cssText = 'font-size:12px;color:#94a3b8;margin-bottom:4px;';
                badge.innerText = '🎤 Audio attached';
                div.appendChild(badge);
            } else if (f.type === 'video') {
                const badge = document.createElement('div');
                badge.style.cssText = 'font-size:12px;color:#94a3b8;margin-bottom:4px;';
                badge.innerText = '🎬 Video attached';
                div.appendChild(badge);
            }
        });
    }

    const textSpan = document.createElement('span');
    textSpan.innerText = text;
    div.appendChild(textSpan);

    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return div;
}

function renderContent(element, content) {
    element.innerHTML = '';

    // Parse thinking block: <|channel>thought\n...<channel|>
    const thinkRegex = /<\|channel>thought\n([\s\S]*?)<channel\|>/;
    const thinkMatch = content.match(thinkRegex);
    const partialThinkStart = content.includes('<|channel>thought\n');
    const partialThinkEnd = content.includes('<channel|>');

    if (thinkMatch) {
        // Complete thinking block
        const thoughtDiv = document.createElement('div');
        thoughtDiv.className = 'thought-block';
        thoughtDiv.innerText = '💭 ' + thinkMatch[1].trim();
        element.appendChild(thoughtDiv);

        const answer = content.replace(thinkMatch[0], '').trim();
        if (answer) {
            const textSpan = document.createElement('span');
            textSpan.innerText = answer;
            element.appendChild(textSpan);
        }
    } else if (partialThinkStart && !partialThinkEnd) {
        // Thinking in progress
        const rawThought = content.replace('<|channel>thought\n', '');
        const thoughtDiv = document.createElement('div');
        thoughtDiv.className = 'thought-block';
        thoughtDiv.innerText = '💭 Thinking...\n' + rawThought;
        element.appendChild(thoughtDiv);
    } else {
        const textSpan = document.createElement('span');
        textSpan.innerText = content;
        element.appendChild(textSpan);
    }

    const chatContainer = document.getElementById('chatMessages');
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

/* ====================== Utilities ====================== */

function clearChat() {
    chatHistory = [];
    document.getElementById('chatMessages').innerHTML =
        '<div class="ai-message message">Chat cleared. How can I help?</div>';
}
