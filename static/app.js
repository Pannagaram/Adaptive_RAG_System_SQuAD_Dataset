/* ══════════════════════════════════════════════════════════════
   Adaptive RAG — Gemini Theme Logic
   ══════════════════════════════════════════════════════════════ */

const API_BASE = "";
let sessionHistory = [];
let currentSessionId = null;
const queryInput = document.getElementById("queryInput");
const queryBtn = document.getElementById("queryBtn");
const messagesContainer = document.getElementById("messagesContainer");
const welcomeScreen = document.getElementById("welcomeScreen");
const chatContainer = document.getElementById("chatContainer");

// Auto-resize textarea
queryInput.addEventListener("input", function() {
    this.style.height = "auto";
    this.style.height = (this.scrollHeight) + "px";
    
    if (this.value.trim().length > 0) {
        queryBtn.classList.add("active");
    } else {
        queryBtn.classList.remove("active");
    }
});

queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleQuery();
    }
});

queryBtn.addEventListener("click", handleQuery);

function setQuery(text) {
    queryInput.value = text;
    queryInput.style.height = "auto";
    queryInput.style.height = (queryInput.scrollHeight) + "px";
    queryBtn.classList.add("active");
    queryInput.focus();
}

async function handleQuery() {
    const query = queryInput.value.trim();
    if (!query) return;

    // Hide welcome screen
    welcomeScreen.style.display = "none";

    // Clear input
    queryInput.value = "";
    queryInput.style.height = "auto";
    queryBtn.classList.remove("active");

    // Add User Message
    appendMessage(query, "user");

    // Add Loading AI Message
    const loadingId = appendLoadingMessage();
    scrollToBottom();

    try {
        const res = await fetch(`${API_BASE}/api/query`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query, verbose: true, history: sessionHistory }),
        });

        if (!res.ok) throw new Error("Request failed");

        const data = await res.json();
        
        // Add current turn to session history
        sessionHistory.push({ role: "user", text: query });
        sessionHistory.push({ role: "ai", text: data.answer });
        
        // Replace loading message with actual answer
        replaceLoadingWithMessage(loadingId, data);
        
        // Only add to sidebar if it's the first query of the session
        if (!currentSessionId) {
            currentSessionId = Date.now().toString();
            addToSidebar(query, currentSessionId);
        }

    } catch (err) {
        replaceLoadingWithError(loadingId, err.message);
    }
    
    scrollToBottom();
}

function appendMessage(text, role) {
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${role}`;
    
    const avatar = role === "user" ? "user" : "ai";
    const avatarContent = role === "user" ? "U" : "✨";
    
    msgDiv.innerHTML = `
        <div class="avatar ${avatar}">${avatarContent}</div>
        <div class="message-content">
            <div class="message-text">${text}</div>
        </div>
    `;
    
    messagesContainer.appendChild(msgDiv);
}

function appendLoadingMessage() {
    const id = "msg-" + Date.now();
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ai`;
    msgDiv.id = id;
    
    msgDiv.innerHTML = `
        <div class="avatar ai loading"></div>
        <div class="message-content">
            <div class="message-text" style="color: var(--text-secondary)">Thinking...</div>
        </div>
    `;
    
    messagesContainer.appendChild(msgDiv);
    return id;
}

function replaceLoadingWithMessage(id, data) {
    const msgDiv = document.getElementById(id);
    if (!msgDiv) return;
    
    // Remove loading class from avatar
    msgDiv.querySelector('.avatar').classList.remove('loading');
    msgDiv.querySelector('.avatar').innerHTML = '✨';

    // Build Sources HTML
    let sourcesHtml = '';
    if (data.sources && data.sources.length > 0) {
        sourcesHtml = data.sources.map(src => `
            <div class="source-item">
                <strong>${src.title || 'Document'}</strong>
                <span>${src.text}</span>
            </div>
        `).join("");
    }

    // Build the detailed answer content
    msgDiv.querySelector('.message-content').innerHTML = `
        <div class="message-text">${data.answer}</div>
        
        <div class="answer-details">
            <div class="details-header" onclick="this.nextElementSibling.classList.toggle('open')">
                <span class="material-icons">info</span>
                Analysis & Metrics (Latency: ${data.total_latency}s)
                <span class="material-icons" style="margin-left: auto;">expand_more</span>
            </div>
            <div class="details-content">
                <div class="metrics-row">
                    <div class="metric">
                        <span class="metric-lbl">Confidence</span>
                        <span class="metric-val">${(data.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-lbl">Top-K</span>
                        <span class="metric-val">${data.top_k_used}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-lbl">Strategy</span>
                        <span class="metric-val">${data.strategy_used.toUpperCase()}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-lbl">Re-rank</span>
                        <span class="metric-val">${data.used_reranking ? "ON" : "OFF"}</span>
                    </div>
                </div>
                
                <div class="adaptive-reason">
                    <strong>Adaptive Reasoning:</strong><br>
                    ${data.adaptive_reason}
                </div>
                
                ${sourcesHtml ? `
                <div style="margin-top: 16px; font-size: 0.85rem; font-weight: 500; color: var(--text-secondary)">
                    Sources used:
                </div>
                <div class="sources-list">${sourcesHtml}</div>
                ` : ''}
            </div>
        </div>
    `;
}

function replaceLoadingWithError(id, errorMsg) {
    const msgDiv = document.getElementById(id);
    if (!msgDiv) return;
    
    msgDiv.querySelector('.avatar').classList.remove('loading');
    msgDiv.querySelector('.avatar').innerHTML = '⚠️';
    
    msgDiv.querySelector('.message-content').innerHTML = `
        <div class="message-text" style="color: #f43f5e">
            Sorry, I encountered an error: ${errorMsg}
        </div>
    `;
}

function scrollToBottom() {
    chatContainer.scrollTo({
        top: chatContainer.scrollHeight,
        behavior: 'smooth'
    });
}

// ── Sidebar History ──────────────────────────────────────────
function addToSidebar(title, sessionId) {
    const list = document.getElementById("historyList");
    const item = document.createElement("div");
    item.className = "history-item active";
    item.dataset.sessionId = sessionId;
    
    // Truncate title
    const shortTitle = title.length > 30 ? title.substring(0, 27) + "..." : title;
    
    item.innerHTML = `
        <span class="material-icons" style="font-size: 18px">chat_bubble_outline</span>
        <span class="history-text">${shortTitle}</span>
    `;
    
    // Clear other active classes
    document.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));
    
    item.onclick = function() {
        document.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));
        this.classList.add('active');
        // A real app would load the specific chat history here
    };
    list.prepend(item);
}

function startNewChat() {
    // Reset session state
    sessionHistory = [];
    currentSessionId = null;
    
    // Clear chat messages UI
    messagesContainer.innerHTML = "";
    
    // Show welcome screen again
    welcomeScreen.style.display = "block";
    
    // Clear active class from sidebar items
    document.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));
}

// Check pipeline status on load
document.addEventListener("DOMContentLoaded", async () => {
    try {
        const res = await fetch(`${API_BASE}/api/status`);
        const data = await res.json();
        if (data.state === "ready" || data.state === "not_started") {
            document.getElementById("loadingOverlay").classList.add("hidden");
        } else {
            document.getElementById("loadingText").textContent = data.message;
            setTimeout(() => location.reload(), 3000);
        }
    } catch {
        document.getElementById("loadingOverlay").classList.add("hidden");
    }
});
