// Puzzel — AI Chat Frontend JavaScript

const API_BASE = window.location.origin;
const GENERATE_ENDPOINT = `${API_BASE}/generate`;
const HEALTH_ENDPOINT = `${API_BASE}/health`;

const messageInput = document.getElementById("messageInput");
const chatForm = document.getElementById("chatForm");
const sendBtn = document.getElementById("sendBtn");
const stopBtn = document.getElementById("stopBtn");
const regenerateBtn = document.getElementById("regenerateBtn");
const messagesContainer = document.getElementById("messages");
const typingIndicator = document.getElementById("typingIndicator");
const charCount = document.getElementById("charCount");
const tempSlider = document.getElementById("tempSlider");
const tempValue = document.getElementById("tempValue");
const maxTokensSlider = document.getElementById("maxTokensSlider");
const maxTokensValue = document.getElementById("maxTokensValue");
const settingsModal = document.getElementById("settingsModal");
const settingsBtn = document.getElementById("settingsBtn");
const closeSettings = document.getElementById("closeSettings");
const newChatBtn = document.getElementById("newChatBtn");

let abortController = null;
let isGenerating = false;

// Character counter
messageInput.addEventListener("input", () => {
    charCount.textContent = messageInput.value.length;
    autoResize(messageInput);
});

function autoResize(textarea) {
    textarea.style.height = "auto";
    textarea.style.height = textarea.scrollHeight + "px";
}

// Send message
chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const text = messageInput.value.trim();
    if (!text || isGenerating) return;

    messageInput.value = "";
    charCount.textContent = "0";
    messageInput.style.height = "auto";

    // Add user message
    appendMessage(text, "user");

    // Show typing
    setTyping(true);
    isGenerating = true;
    sendBtn.disabled = true;
    stopBtn.style.display = "flex";
    regenerateBtn.style.display = "none";

    abortController = new AbortController();

    try {
        const response = await fetch(GENERATE_ENDPOINT, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                prompt: text,
                max_new_tokens: parseInt(maxTokensSlider.value, 10),
                temperature: parseFloat(tempSlider.value),
                stop_at_eos: true,
            }),
            signal: abortController.signal,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }

        const data = await response.json();

        // Remove typing indicator and add AI message
        setTyping(false);
        appendMessage(data.generated_text, "ai");
    } catch (err) {
        setTyping(false);
        if (err.name === "AbortError") {
            // User stopped generation — don't show error
            return;
        }
        appendMessage(`Error: ${err.message}`, "ai");
    } finally {
        isGenerating = false;
        sendBtn.disabled = false;
        stopBtn.style.display = "none";
        regenerateBtn.style.display = "flex";
        abortController = null;
    }
});

// Stop generation
stopBtn.addEventListener("click", () => {
    if (abortController) {
        abortController.abort();
    }
    isGenerating = false;
    setTyping(false);
    sendBtn.disabled = false;
    stopBtn.style.display = "none";
    regenerateBtn.style.display = "flex";
});

// Regenerate last message
regenerateBtn.addEventListener("click", () => {
    const lastUserMsg = messagesContainer.querySelector(
        ".message.user-message:last-child .message-content"
    );
    if (!lastUserMsg) return;

    const text = lastUserMsg.textContent;
    messageInput.value = text;
    charCount.textContent = text.length;

    // Remove the last AI message (and user message if we want to redo)
    const aiMsg = messagesContainer.querySelector(".message.ai-message:last-child");
    if (aiMsg) aiMsg.remove();

    chatForm.requestSubmit();
});

// Append a message to the chat
function appendMessage(text, role) {
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${role}-message`;

    const contentDiv = document.createElement("div");
    contentDiv.className = "message-content";
    contentDiv.textContent = text;

    msgDiv.appendChild(contentDiv);
    messagesContainer.appendChild(msgDiv);
    scrollToBottom();
}

// Show/hide typing indicator
function setTyping(show) {
    typingIndicator.style.display = show ? "flex" : "none";
    if (show) scrollToBottom();
}

// Scroll to bottom
function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Settings modal
settingsBtn.addEventListener("click", () => {
    settingsModal.classList.add("show");
});

closeSettings.addEventListener("click", () => {
    settingsModal.classList.remove("show");
});

settingsModal.addEventListener("click", (e) => {
    if (e.target === settingsModal) {
        settingsModal.classList.remove("show");
    }
});

tempSlider.addEventListener("input", () => {
    tempValue.textContent = tempSlider.value;
});

maxTokensSlider.addEventListener("input", () => {
    maxTokensValue.textContent = maxTokensSlider.value;
});

// New conversation
newChatBtn.addEventListener("click", () => {
    messagesContainer.innerHTML = "";
    appendMessage(
        "Hello! I'm Puzzel, a multilingual language model. How can I help you today?",
        "ai"
    );
    regenerateBtn.style.display = "none";
    stopBtn.style.display = "none";
    sendBtn.disabled = false;
});

// Check server health on load
async function checkHealth() {
    try {
        const response = await fetch(HEALTH_ENDPOINT);
        const data = await response.json();
        console.log("Server health:", data);
    } catch (err) {
        console.warn("Cannot reach API server:", err.message);
        // Show offline notice
        const notice = document.createElement("div");
        notice.style.cssText =
            "position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); " +
            "background: #ef4444; color: white; padding: 8px 16px; border-radius: 6px; " +
            "font-size: 0.875rem; z-index: 300;";
        notice.textContent = "Cannot connect to API server. Start the backend first.";
        document.body.appendChild(notice);
        setTimeout(() => notice.remove(), 5000);
    }
}

checkHealth();