import os
import sys
import re
import streamlit as st

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import CustomerSupportPipeline

st.set_page_config(
    page_title="AntiGravity AI | Support Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- High-End AntiGravity Dark Theme & Modern Typography CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Global Reset & Dark Theme */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #0b0f19;
    color: #e2e8f0;
}

/* Hide default Streamlit header and footer */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #0b0f19;
}
::-webkit-scrollbar-thumb {
    background: #1e293b;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #334155;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #080c14 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.07);
}

section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 {
    color: #f8fafc !important;
    font-weight: 700;
    letter-spacing: -0.02em;
}

/* Antigravity Header Badge */
.ag-header {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
}

.ag-title {
    font-size: 26px;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 0;
}

.ag-subtitle {
    color: #94a3b8;
    font-size: 13.5px;
    margin-top: 6px;
    margin-bottom: 0;
    letter-spacing: 0.01em;
}

.ag-badges-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 14px;
}

.ag-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 11.5px;
    font-weight: 600;
    letter-spacing: 0.02em;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: #cbd5e1;
}

.ag-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
}
.dot-emerald { background: #10b981; box-shadow: 0 0 8px #10b981; }
.dot-cyan { background: #06b6d4; box-shadow: 0 0 8px #06b6d4; }
.dot-violet { background: #8b5cf6; box-shadow: 0 0 8px #8b5cf6; }
.dot-rose { background: #f43f5e; box-shadow: 0 0 8px #f43f5e; }

/* Scenario Preset Buttons in Sidebar */
.stButton > button {
    background: #111827 !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    text-align: left !important;
    display: flex !important;
    justify-content: flex-start !important;
    width: 100% !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.15), rgba(129, 140, 248, 0.15)) !important;
    border-color: rgba(99, 102, 241, 0.5) !important;
    color: #38bdf8 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(56, 189, 248, 0.15) !important;
}

/* Chat Message Bubbles */
.chat-bubble-container {
    display: flex;
    gap: 14px;
    margin-bottom: 20px;
    align-items: flex-start;
}

.chat-avatar {
    width: 38px;
    height: 38px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

.avatar-user {
    background: linear-gradient(135deg, #3b82f6, #1d4ed8);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.avatar-assistant {
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 0 16px rgba(99, 102, 241, 0.3);
}

.chat-content-box {
    flex-grow: 1;
    background: rgba(17, 24, 39, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.07);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 16px 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    position: relative;
    max-width: calc(100% - 52px);
}

.chat-content-user {
    background: rgba(26, 36, 56, 0.85);
    border: 1px solid rgba(56, 189, 248, 0.2);
}

.text-rtl {
    direction: rtl !important;
    text-align: right !important;
    font-family: 'Cairo', 'Tajawal', sans-serif !important;
    font-size: 15px !important;
    line-height: 1.85 !important;
    color: #f1f5f9;
}

.text-ltr {
    direction: ltr !important;
    text-align: left !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14.5px !important;
    line-height: 1.65 !important;
    color: #f1f5f9;
}

/* Telemetry Card Styling */
.telemetry-card {
    background: rgba(10, 15, 26, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 16px;
    margin-top: 14px;
}

.telemetry-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 10px;
    margin-bottom: 12px;
}

.telemetry-stat {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 10px 12px;
}

.stat-label {
    font-size: 10.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748b;
    margin-bottom: 4px;
}

.stat-val {
    font-size: 13.5px;
    font-weight: 700;
    color: #f8fafc;
}

/* Chat Input Bar */
[data-testid="stChatInput"] {
    background-color: #0b0f19 !important;
}

[data-testid="stChatInput"] > div {
    background-color: #111827 !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4) !important;
    transition: all 0.2s ease;
}

[data-testid="stChatInput"] > div:focus-within {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25), 0 8px 30px rgba(0, 0, 0, 0.5) !important;
}

[data-testid="stChatInput"] textarea {
    color: #f8fafc !important;
    font-family: 'Inter', 'Cairo', sans-serif !important;
    font-size: 14.5px !important;
}

/* Expander custom styling */
.streamlit-expanderHeader {
    background: rgba(255, 255, 255, 0.02) !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-size: 12.5px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# Cache pipeline loading
@st.cache_resource(show_spinner="Booting AntiGravity Neural Engine & FAISS Vector Index...")
def load_pipeline():
    return CustomerSupportPipeline(models_dir="models")

pipeline = load_pipeline()

def is_arabic(text):
    return bool(re.search(r'[\u0600-\u06FF]', text))

# --- AntiGravity Brand Header ---
st.markdown("""
<div class="ag-header">
    <div class="ag-title">
        <span>⚡ AntiGravity RAG Support Agent</span>
    </div>
    <p class="ag-subtitle">
        Enterprise-grade multi-stage NLP pipeline • Real-time intent routing • Sentiment empathy • FAISS vector retrieval
    </p>
    <div class="ag-badges-row">
        <span class="ag-pill"><span class="ag-dot dot-emerald"></span> System Online</span>
        <span class="ag-pill"><span class="ag-dot dot-cyan"></span> Groq GPT-OSS-120B</span>
        <span class="ag-pill"><span class="ag-dot dot-violet"></span> FAISS 26,778 Vectors</span>
        <span class="ag-pill"><span class="ag-dot dot-rose"></span> Bi-LSTM Sentiment Active</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your AntiGravity intelligent customer support assistant. How can I assist you with your orders, refunds, shipments, or account today?",
            "telemetry": None
        }
    ]

# Sidebar Scenarios & Controls
st.sidebar.markdown("""
<div style="padding: 10px 0 16px 0;">
    <h3 style="margin:0; font-size: 16px; color: #f8fafc;">🧪 Neural Scenarios</h3>
    <p style="margin: 4px 0 0 0; font-size: 12px; color: #64748b;">Instant testing across pipeline branches:</p>
</div>
""", unsafe_allow_html=True)

presets = [
    ("🇪🇬 استرجاع لمنتج مكسور", "استلمت المنتج مكسور وعايز فلوسي ترجع"),
    ("👋 تحية بالعربية", "السلام عليكم، كيف حالك؟"),
    ("📦 Order Tracking (EN)", "Where is my order #55231 and when will it arrive?"),
    ("😠 Angry Refund Dispute (EN)", "I received a damaged item yesterday and nobody is answering! I demand a full refund right now!"),
    ("🔑 Account Password Reset (EN)", "I forgot my account password, how can I recover it?"),
    ("👋 Greeting & Smalltalk (EN)", "Hello, good morning!"),
    ("🌌 Out of Scope Trivia (EN)", "What is the distance between the Earth and the Moon?")
]

selected_preset = None
for label, query in presets:
    if st.sidebar.button(label, use_container_width=True):
        selected_preset = query

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 11.5px; color: #64748b; line-height: 1.6;">
    <strong style="color: #cbd5e1;">Pipeline Architecture:</strong><br>
    1. Language Detector (20 langs, SGD)<br>
    2. Sentiment Classifier (Bi-LSTM)<br>
    3. Intent Router (7 Condensed Classes)<br>
    4. Grounded Q&A RAG (FAISS + Groq)
</div>
""", unsafe_allow_html=True)

if st.sidebar.button("🗑️ Clear Conversation", use_container_width=True):
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Conversation cleared. How can I assist you today?",
            "telemetry": None
        }
    ]
    st.rerun()

def format_chat_text(text, is_rtl=False):
    # Convert bold **text** to styled <strong>
    formatted = re.sub(r'\*\*(.*?)\*\*', r'<strong style="color: #38bdf8; font-weight: 700;">\1</strong>', text)
    lines = formatted.split('\n')
    processed = []
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            processed.append('<div style="height: 6px;"></div>')
        elif re.match(r'^\d+[\.\)]', line_clean):
            prefix = line_clean.split()[0]
            rest = line_clean[len(prefix):].strip()
            pad = "padding-right: 16px;" if is_rtl else "padding-left: 16px;"
            processed.append(f'<div style="margin: 4px 0; {pad}"><span style="color: #818cf8; font-weight: 700;">{prefix}</span> {rest}</div>')
        elif line_clean.startswith('- ') or line_clean.startswith('• '):
            pad = "padding-right: 16px;" if is_rtl else "padding-left: 16px;"
            processed.append(f'<div style="margin: 4px 0; {pad}"><span style="color: #38bdf8;">•</span> {line_clean[2:].strip()}</div>')
        elif line_clean.startswith('[') and line_clean.endswith(']'):
            processed.append(f'<div style="background: rgba(244,63,94,0.12); border-left: 3px solid #f43f5e; padding: 6px 12px; border-radius: 6px; font-weight: 600; color: #f43f5e; margin: 8px 0;">{line_clean}</div>')
        else:
            processed.append(f'<div style="margin-bottom: 4px;">{line_clean}</div>')
    return "".join(processed)

# --- Render Chat History ---
for idx, msg in enumerate(st.session_state.messages):
    is_user = msg["role"] == "user"
    avatar_class = "avatar-user" if is_user else "avatar-assistant"
    avatar_icon = "👤" if is_user else "⚡"
    box_class = "chat-content-user" if is_user else ""
    rtl = is_arabic(msg["content"])
    text_dir_class = "text-rtl" if rtl else "text-ltr"
    
    formatted_html = format_chat_text(msg["content"], is_rtl=rtl)
    
    chat_html = f"""
    <div class="chat-bubble-container">
        <div class="chat-avatar {avatar_class}">{avatar_icon}</div>
        <div class="chat-content-box {box_class}">
            <div class="{text_dir_class}">{formatted_html}</div>
        </div>
    </div>
    """
    st.markdown(chat_html, unsafe_allow_html=True)
    
    # If telemetry available, render modern inspection card
    if msg.get("telemetry"):
        tel = msg["telemetry"]
        is_last_msg = (idx == len(st.session_state.messages) - 1)
        with st.expander("🔍 Real-time Pipeline Telemetry & Routing Details", expanded=is_last_msg):
            sent = tel['detected_sentiment']
            sent_badge = "🔴 Negative" if sent == "negative" else ("🟢 Positive" if sent == "positive" else "⚪ Neutral")
            esc_badge = "⚠️ URGENT (Priority Human Flag)" if tel['priority_escalation'] else "✔️ Standard Automated"
            
            st.markdown(f"""
            <div class="telemetry-card">
                <div class="telemetry-grid">
                    <div class="telemetry-stat">
                        <div class="stat-label">Language</div>
                        <div class="stat-val" style="color: #38bdf8;">{tel['detected_language'].upper()} ({tel['language_confidence']:.1%})</div>
                    </div>
                    <div class="telemetry-stat">
                        <div class="stat-label">Sentiment</div>
                        <div class="stat-val">{sent_badge}</div>
                    </div>
                    <div class="telemetry-stat">
                        <div class="stat-label">Intent</div>
                        <div class="stat-val" style="color: #a855f7;">{tel['detected_intent']}</div>
                    </div>
                    <div class="telemetry-stat">
                        <div class="stat-label">Escalation</div>
                        <div class="stat-val" style="color: {'#f43f5e' if tel['priority_escalation'] else '#10b981'};">{esc_badge}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"**Routing Pathway:** `{tel['routing_action']}`")
            if tel.get("translated_query"):
                st.markdown(f"**Normalized English Query (Knowledge-Base Search):** `{tel['translated_query']}`")
            
            if tel.get("retrieved_chunks"):
                st.markdown("**Grounded Knowledge-Base Documents (FAISS Cosine Similarity):**")
                for c_idx, chk in enumerate(tel["retrieved_chunks"], 1):
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.02); border-left: 3px solid #6366f1; padding: 8px 12px; margin-bottom: 8px; border-radius: 4px;">
                        <span style="color: #818cf8; font-size: 11px; font-weight: 600;">DOC #{c_idx} | Similarity: {chk['score']:.3f} | Intent: {chk.get('intent', 'N/A')}</span>
                        <div style="font-size: 13px; color: #cbd5e1; margin-top: 4px;">{chk['response']}</div>
                    </div>
                    """, unsafe_allow_html=True)

# --- Chat Input & Generation Flow ---
user_input = st.chat_input("Type your message in English, Arabic, or any language...")
if selected_preset:
    user_input = selected_preset

if user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input, "telemetry": None})

    # Process with pipeline
    with st.spinner("Processing through AntiGravity 4-stage pipeline..."):
        result = pipeline.process_message(user_input)
        
        # Store assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["response"],
            "telemetry": result
        })
    st.rerun()
