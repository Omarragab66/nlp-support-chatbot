import os
import sys
import streamlit as st

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import CustomerSupportPipeline

st.set_page_config(
    page_title="E-commerce Customer Support Chatbot",
    page_icon="🛍️",
    layout="wide"
)

# Cache pipeline loading
@st.cache_resource(show_spinner="Loading NLP Models and FAISS Vector Store...")
def load_pipeline():
    return CustomerSupportPipeline(models_dir="models")

pipeline = load_pipeline()

# Title and Header
st.title("🛍️ RAG-Based E-commerce Support Chatbot")
st.caption("Powered by Language Detection • Bi-LSTM Sentiment • Intent Classifier • FAISS Vector Store • Groq LLM (GPT-OSS-120B)")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your automated customer support assistant. How can I assist you with your orders, refunds, deliveries, or account today?",
            "telemetry": None
        }
    ]

# Preset queries for quick testing
st.sidebar.title("🧪 Quick Test Scenarios")
st.sidebar.caption("Click any query to test pipeline routing:")

presets = [
    ("👋 Greeting", "Hello, good morning!"),
    ("📦 Order Status", "Where is my order #55231 and when will it arrive?"),
    ("😠 Frustrated Refund", "I received a damaged item yesterday and nobody is answering! I demand a full refund right now!"),
    ("🔑 Password Reset", "I forgot my account password, how can I recover it?"),
    ("🌌 Out of Scope", "What is the speed of light in vacuum?")
]

selected_preset = None
for label, query in presets:
    if st.sidebar.button(label, use_container_width=True):
        selected_preset = query

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # If assistant has telemetry, show expandable metadata
        if msg.get("telemetry"):
            tel = msg["telemetry"]
            with st.expander("🔍 Real-time Pipeline Telemetry & Routing Details"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Language", f"{tel['detected_language'].upper()} ({tel['language_confidence']:.1%})")
                with col2:
                    sent = tel['detected_sentiment']
                    color = "🔴" if sent == "negative" else ("🟢" if sent == "positive" else "⚪")
                    st.metric("Sentiment", f"{color} {sent.capitalize()}")
                with col3:
                    st.metric("Intent", tel['detected_intent'])
                with col4:
                    esc = "⚠️ YES (Urgent)" if tel['priority_escalation'] else "✔️ Normal"
                    st.metric("Escalation", esc)
                    
                st.markdown(f"**Routing Action:** `{tel['routing_action']}`")
                
                if tel.get("retrieved_chunks"):
                    st.markdown("**Retrieved Knowledge Base Chunks (FAISS):**")
                    for i, chk in enumerate(tel["retrieved_chunks"], 1):
                        st.markdown(f"**Chunk {i} (Similarity: {chk['score']:.3f} | Intent: `{chk.get('intent', 'N/A')}`):**")
                        st.info(chk["response"])

# Handle input
user_input = st.chat_input("Type your message here...")
if selected_preset:
    user_input = selected_preset

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input, "telemetry": None})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process with unified pipeline
    with st.chat_message("assistant"):
        with st.spinner("Processing through 4-stage pipeline..."):
            result = pipeline.process_message(user_input)
            st.markdown(result["response"])
            
            # Show telemetry
            with st.expander("🔍 Real-time Pipeline Telemetry & Routing Details", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Language", f"{result['detected_language'].upper()} ({result['language_confidence']:.1%})")
                with col2:
                    sent = result['detected_sentiment']
                    color = "🔴" if sent == "negative" else ("🟢" if sent == "positive" else "⚪")
                    st.metric("Sentiment", f"{color} {sent.capitalize()}")
                with col3:
                    st.metric("Intent", result['detected_intent'])
                with col4:
                    esc = "⚠️ YES (Urgent)" if result['priority_escalation'] else "✔️ Normal"
                    st.metric("Escalation", esc)
                    
                st.markdown(f"**Routing Action:** `{result['routing_action']}`")
                
                if result.get("retrieved_chunks"):
                    st.markdown("**Retrieved Knowledge Base Chunks (FAISS):**")
                    for i, chk in enumerate(result["retrieved_chunks"], 1):
                        st.markdown(f"**Chunk {i} (Similarity: {chk['score']:.3f} | Intent: `{chk.get('intent', 'N/A')}`):**")
                        st.info(chk["response"])

    # Store in history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["response"],
        "telemetry": result
    })
