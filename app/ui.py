import os
import sys
import re
import textwrap
import streamlit as st

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import CustomerSupportPipeline

st.set_page_config(
    page_title="Customer Support",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Strict Black & White Theme CSS (Minimalist Retail Support Widget) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* Global Reset: Solid Black Theme */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #0a0a0a !important;
    color: #e5e5e5 !important;
}

/* Hide Streamlit default header/footer */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* Custom Minimal Scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #0a0a0a;
}
::-webkit-scrollbar-thumb {
    background: #262626;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #404040;
}

/* Sidebar Styling: Clean Dark Panel for Evaluation */
section[data-testid="stSidebar"] {
    background-color: #111111 !important;
    border-right: 1px solid #262626 !important;
}

section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
}

/* Main Header */
.store-header {
    border-bottom: 1px solid #262626;
    padding-bottom: 16px;
    margin-bottom: 24px;
}

.store-title {
    font-size: 20px;
    font-weight: 600;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.02em;
}

.store-subtitle {
    font-size: 13px;
    color: #888888;
    margin-top: 4px;
    margin-bottom: 0;
}

/* Clean Message Container */
.msg-wrapper {
    display: flex;
    flex-direction: column;
    margin-bottom: 18px;
    max-width: 82%;
}

.msg-wrapper-user {
    margin-left: auto;
    align-items: flex-end;
}

.msg-wrapper-assistant {
    margin-right: auto;
    align-items: flex-start;
}

.msg-sender {
    font-size: 11px;
    font-weight: 600;
    color: #737373;
    margin-bottom: 5px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.msg-bubble {
    padding: 14px 18px;
    border-radius: 12px;
    font-size: 14.5px;
    line-height: 1.65;
    word-break: break-word;
}

.msg-bubble-user {
    background-color: #262626;
    color: #ffffff;
    border: 1px solid #333333;
}

.msg-bubble-assistant {
    background-color: #141414;
    color: #e5e5e5;
    border: 1px solid #262626;
}

/* RTL Support for Arabic Messages */
.text-rtl {
    direction: rtl !important;
    text-align: right !important;
    font-family: 'Cairo', sans-serif !important;
    font-size: 15px !important;
    line-height: 1.8 !important;
}

.text-ltr {
    direction: ltr !important;
    text-align: left !important;
    font-family: 'Inter', sans-serif !important;
}

/* Escalation Notice inside message */
.escalation-notice {
    margin-top: 10px;
    padding: 8px 12px;
    background-color: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 6px;
    font-size: 12.5px;
    color: #f87171;
    font-weight: 500;
}

/* Sidebar Test Buttons */
.stButton > button {
    background: #171717 !important;
    color: #e5e5e5 !important;
    border: 1px solid #262626 !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    text-align: left !important;
    width: 100% !important;
    transition: background 0.15s ease, border-color 0.15s ease !important;
}

.stButton > button:hover {
    background: #222222 !important;
    border-color: #404040 !important;
    color: #ffffff !important;
}

/* Sidebar Diagnostic Cards */
.diag-card {
    background: #171717;
    border: 1px solid #262626;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 10px;
}

.diag-field {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid #222222;
    font-size: 12px;
}

.diag-field:last-child {
    border-bottom: none;
}

.diag-key {
    color: #888888;
}

.diag-val {
    color: #ffffff;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11.5px;
}

.diag-val-alert {
    color: #f87171;
    font-weight: 600;
}

/* Chat Input Styling */
[data-testid="stChatInput"] {
    background-color: #0a0a0a !important;
}

[data-testid="stChatInput"] > div {
    background-color: #141414 !important;
    border: 1px solid #262626 !important;
    border-radius: 10px !important;
    box-shadow: none !important;
}

[data-testid="stChatInput"] > div:focus-within {
    border-color: #525252 !important;
}

[data-testid="stChatInput"] textarea {
    color: #ffffff !important;
    font-family: 'Inter', 'Cairo', sans-serif !important;
    font-size: 14px !important;
}

/* Streamlit Expander in Sidebar */
.streamlit-expanderHeader {
    background: #171717 !important;
    border: 1px solid #262626 !important;
    border-radius: 6px !important;
    color: #cccccc !important;
    font-size: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# Cache pipeline loading
@st.cache_resource(show_spinner="Initializing support engine...")
def load_pipeline():
    return CustomerSupportPipeline(models_dir="models")

pipeline = load_pipeline()

def is_arabic(text):
    return bool(re.search(r'[\u0600-\u06FF]', text))

def clean_message_content(text):
    """Separate system notice from the main text for clean styling."""
    notice = None
    notice_match = re.search(r'(\[(?:System Notice|إشعار النظام)\]:?[^\n]+)', text, re.IGNORECASE)
    if notice_match:
        notice = notice_match.group(1).strip()
        text = text.replace(notice_match.group(0), "").strip()
    return text, notice

def render_text_lines(text, is_rtl=False):
    """Format paragraphs, numbered steps, and bullet lists cleanly."""
    # Convert bold **text** to strong
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong style="color: #ffffff;">\1</strong>', text)
    lines = text.split('\n')
    rendered = []
    for line in lines:
        l = line.strip()
        if not l:
            rendered.append('<div style="height: 6px;"></div>')
        elif re.match(r'^\d+[\.\)]', l):
            num = l.split()[0]
            rest = l[len(num):].strip()
            pad = "padding-right: 14px;" if is_rtl else "padding-left: 14px;"
            rendered.append(f'<div style="margin: 3px 0; {pad}"><span style="color: #a3a3a3; font-weight: 600;">{num}</span> {rest}</div>')
        elif l.startswith('- ') or l.startswith('• '):
            pad = "padding-right: 14px;" if is_rtl else "padding-left: 14px;"
            rendered.append(f'<div style="margin: 3px 0; {pad}"><span style="color: #737373;">•</span> {l[2:].strip()}</div>')
        else:
            rendered.append(f'<div style="margin-bottom: 4px;">{l}</div>')
    return "".join(rendered)

# --- Customer-Facing Header ---
st.markdown("""
<div class="store-header">
    <div class="store-title">Customer Support</div>
    <div class="store-subtitle">Help with orders, shipping, refunds, and account assistance</div>
</div>
""", unsafe_allow_html=True)

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! How can I help you today with your order, refund, delivery, or account?",
            "telemetry": None
        }
    ]

# --- Sidebar: Developer / Evaluation Panel ONLY ---
st.sidebar.markdown("### Developer / Evaluation Panel")
st.sidebar.caption("Evaluation controls and pipeline inspection telemetry.")

st.sidebar.markdown("**Test Scenarios**")
scenarios = [
    ("Order Status Tracking", "Where is my order #55231 and when will it arrive?"),
    ("Damaged Item & Refund (EN)", "I received a damaged item yesterday and nobody is answering! I demand a full refund right now!"),
    ("استرجاع لمنتج تالف (AR)", "استلمت المنتج مكسور وعايز فلوسي ترجع"),
    ("Password Recovery", "I forgot my account password, how can I recover it?"),
    ("Small Talk & Greeting", "Hello, good morning!"),
    ("Out-of-Scope Query", "What is the distance between the Earth and the Moon?")
]

selected_scenario = None
for label, query in scenarios:
    if st.sidebar.button(label, use_container_width=True):
        selected_scenario = query

st.sidebar.markdown("---")

# Find the most recent telemetry from conversation
latest_telemetry = None
for m in reversed(st.session_state.messages):
    if m.get("telemetry"):
        latest_telemetry = m["telemetry"]
        break

st.sidebar.markdown("**Latest Request Diagnostics**")
if latest_telemetry:
    tel = latest_telemetry
    is_escalated = tel.get("priority_escalation", False)

    diag_html = textwrap.dedent(f"""\
    <div class="diag-card">
    <div class="diag-field">
    <span class="diag-key">Detected Language</span>
    <span class="diag-val">{tel['detected_language'].upper()} ({tel['language_confidence']:.1%})</span>
    </div>
    <div class="diag-field">
    <span class="diag-key">Detected Sentiment</span>
    <span class="diag-val {'diag-val-alert' if tel['detected_sentiment'] == 'negative' else ''}">{tel['detected_sentiment'].capitalize()}</span>
    </div>
    <div class="diag-field">
    <span class="diag-key">Classified Intent</span>
    <span class="diag-val">{tel['detected_intent']}</span>
    </div>
    <div class="diag-field">
    <span class="diag-key">Intent Confidence</span>
    <span class="diag-val">{tel.get('intent_confidence', 0.0):.1%}</span>
    </div>
    <div class="diag-field">
    <span class="diag-key">Routing Action</span>
    <span class="diag-val" style="font-size: 10.5px;">{tel['routing_action']}</span>
    </div>
    <div class="diag-field">
    <span class="diag-key">Escalation Status</span>
    <span class="diag-val {'diag-val-alert' if is_escalated else ''}">{'Escalated (Urgent)' if is_escalated else 'Standard'}</span>
    </div>
    </div>
    """)
    st.sidebar.markdown(diag_html, unsafe_allow_html=True)

    if tel.get("translated_query"):
        st.sidebar.markdown(f"**Normalized Query (EN):**\n`{tel['translated_query']}`")

    if tel.get("retrieved_chunks"):
        with st.sidebar.expander("Retrieved KB Documents (FAISS)", expanded=False):
            for i, chunk in enumerate(tel["retrieved_chunks"], 1):
                st.markdown(f"**Doc {i}** (Similarity: `{chunk['score']:.3f}` | Intent: `{chunk.get('intent', 'N/A')}`)")
                st.caption(chunk["response"])
else:
    empty_state_html = textwrap.dedent("""\
    <div style="font-size: 12px; color: #737373; padding: 8px 0;">
    No requests processed yet. Submit a message or pick a test scenario above to inspect pipeline telemetry.
    </div>
    """)
    st.sidebar.markdown(empty_state_html, unsafe_allow_html=True)

st.sidebar.markdown("---")
if st.sidebar.button("Reset Conversation", use_container_width=True):
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! How can I help you today with your order, refund, delivery, or account?",
            "telemetry": None
        }
    ]
    st.rerun()

# --- Main Customer Chat Display (Clean, No AI/Model Details) ---
for msg in st.session_state.messages:
    is_user = msg["role"] == "user"
    sender_label = "You" if is_user else "Store Assistant"
    wrapper_class = "msg-wrapper-user" if is_user else "msg-wrapper-assistant"
    bubble_class = "msg-bubble-user" if is_user else "msg-bubble-assistant"

    body_text, notice_text = clean_message_content(msg["content"])
    rtl = is_arabic(body_text)
    dir_class = "text-rtl" if rtl else "text-ltr"
    rendered_body = render_text_lines(body_text, is_rtl=rtl)

    notice_html = ""
    if notice_text:
        notice_dir = "text-rtl" if is_arabic(notice_text) else "text-ltr"
        notice_html = f'<div class="escalation-notice {notice_dir}">{notice_text}</div>'

    msg_html = textwrap.dedent(f"""\
    <div class="msg-wrapper {wrapper_class}">
    <div class="msg-sender">{sender_label}</div>
    <div class="msg-bubble {bubble_class}">
    <div class="{dir_class}">{rendered_body}</div>
    {notice_html}
    </div>
    </div>
    """)
    st.markdown(msg_html, unsafe_allow_html=True)

# --- Chat Input & Execution ---
user_input = st.chat_input("Type your message here...")
if selected_scenario:
    user_input = selected_scenario

if user_input:
    # 1. Append user message
    st.session_state.messages.append({"role": "user", "content": user_input, "telemetry": None})

    # 2. Process query through pipeline
    with st.spinner("Responding..."):
        result = pipeline.process_message(user_input)

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["response"],
            "telemetry": result
        })
    st.rerun()