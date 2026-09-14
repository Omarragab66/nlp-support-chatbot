# RAG-Based E-commerce Customer Support Chatbot
**NLP Final Task 2026**

An end-to-end intelligent customer support chatbot for an online retailer. The system integrates Language Detection, Deep Learning Sentiment Analysis, Multi-class Intent Routing, and Grounded Retrieval-Augmented Generation (RAG) powered by FAISS and Groq LLM (`openai/gpt-oss-120b`).

---

## Architecture & System Overview

```
                          Customer Message
                                 │
                                 ▼
               ┌───────────────────────────────────┐
               │  1. Language Detection Classifier │ ──► Detects Language (20 languages)
               └───────────────────────────────────┘
                                 │
                                 ▼
               ┌───────────────────────────────────┐
               │ 2. Sentiment Classifier (Bi-LSTM) │ ──► Negative / Neutral / Positive Tone
               └───────────────────────────────────┘
                                 │
                                 ▼
               ┌───────────────────────────────────┐
               │   3. Intent Classifier & Routing  │ ──► Routes user intent (7 classes)
               └───────────────────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
  [greeting_smalltalk]     [out_of_scope]      [Support Inquiries]
  Direct Canned Reply      Polite Boundary     (orders, billing, refunds, accounts)
  (Zero RAG latency)       Offer human rep               │
                                                         ▼
                                               ┌─────────────────────┐
                                               │ 4. FAISS Vector DB  │
                                               │  (Cosine Top-k KB)  │
                                               └─────────────────────┘
                                                         │
                                                         ▼
                                               ┌─────────────────────┐
                                               │   Groq LLM Engine   │
                                               │ (openai/gpt-oss-120b)│
                                               └─────────────────────┘
                                                         │
                                                         ▼
                                               Empathetic Grounded Response
                                               (Priority Flag if Complaint)
```

---

## Key Modules & Technical Decisions

### 1. Language Detection (`notebooks/01_language_detection.ipynb`)
- **Dataset**: `papluca/language-identification` (90,000 samples, 20 languages).
- **Technique**: Traditional NLP with Subword Character n-grams (`analyzer='char_wb'`, range 2-5) + Linear SGD Classifier with modified Huber loss for calibrated probabilities.
- **Why Character n-grams?**: As required by project enhancements, character n-grams capture morphological roots, prefixes, and suffixes, making the model exceptionally resilient to typos, abbreviations, non-Latin scripts (Arabic, Cyrillic, Chinese), and short text queries.
- **Performance**: Test Accuracy **99.59%** (vs. 90.36% word-level baseline).

### 2. Sentiment / Emotion Classifier (`notebooks/02_sentiment_classifier.ipynb`)
- **Dataset**: `dair-ai/emotion` (20k samples across 6 emotions mapped into 3 target classes: Negative, Positive, Neutral).
- **Domain Shift Mitigation**: Augmented with e-commerce support questions and complaints from `bitext` to eliminate the distribution gap between Twitter text and customer support queries.
- **Architecture**: Deep Neural Network in PyTorch with **Bidirectional LSTM**, Word Embeddings, Dropout (0.3), and Global Max Pooling with class-weighted loss.
- **Performance**: Test Accuracy **97.57%**, Macro F1 **0.9689**.

### 3. Intent Classification & Routing (`notebooks/03_intent_classifier.ipynb`)
- **Dataset**: `bitext/Bitext-customer-support-llm-chatbot-training-dataset` (28,702 samples).
- **Intent Consolidation**: Condensed the 27 fine-grained intents into the 7 recommended routing categories:
  1. `greeting_smalltalk`
  2. `order_status`
  3. `order_management`
  4. `billing_and_refunds`
  5. `account_management`
  6. `complaint`
  7. `out_of_scope`
- **Handling `out_of_scope`**: Hybrid approach combining explicit training on non-retail general domain questions and an inference confidence threshold (< 0.35).
- **Performance**: Test Accuracy **99.69%**, Macro F1 **0.9970**.

### 4. Q&A RAG Pipeline (`notebooks/04_rag_pipeline.ipynb`)
- **Knowledge Base**: 26,778 curated instruction-response support documents from `bitext`.
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dimensions).
- **Vector Database**: Local **FAISS** index with Inner Product cosine similarity.
- **LLM Generation**: **Groq API** with model `openai/gpt-oss-120b` running with dynamic system prompts conditioned on the detected customer sentiment (apologetic and empathetic tone when frustrated, priority escalation flag on complaints).

---

## Project Directory Layout

```
nlp-support-chatbot/
│
├── notebooks/
│   ├── 01_language_detection.ipynb      # Language detection training & evaluation
│   ├── 02_sentiment_classifier.ipynb    # BiLSTM sentiment training & evaluation
│   ├── 03_intent_classifier.ipynb       # Intent classifier & routing logic
│   └── 04_rag_pipeline.ipynb            # FAISS vector store & Groq LLM pipeline
│
├── src/
│   └── pipeline.py                      # Unified CustomerSupportPipeline class
│
├── app/
│   ├── api.py                           # FastAPI REST server (/chat, /analyze, /health)
│   └── ui.py                            # Streamlit interactive Web Chat UI
│
├── models/
│   ├── language_detector.pkl            # Trained language detection model
│   ├── sentiment_bilstm.pt              # Trained PyTorch BiLSTM weights
│   ├── sentiment_tokenizer.json         # Vocabulary dictionary for sentiment
│   ├── intent_classifier.pkl            # Trained intent classification pipeline
│   ├── faiss_index.bin                  # Indexed FAISS vector database
│   ├── kb_chunks.json                   # Knowledge base documents
│   └── *.png                            # Confusion matrix evaluation plots
│
├── .env                                 # Environment configuration (GROQ_API_KEY)
├── requirements.txt                     # Python dependencies
└── README.md                            # Comprehensive project documentation
```

---

## Setup & Running the Application

### 1. Environment Setup
```bash
# Clone or navigate to the repository
cd nlp-support-chatbot

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate       # Windows
# source venv/bin/activate    # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Groq API Key
Add your Groq API key to `.env`:
```env
GROQ_API_KEY=gsk_...
```

### 3. Run the Interactive Web UI (Streamlit)
To launch the interactive chat interface with real-time pipeline telemetry:
```bash
streamlit run app/ui.py
```
Open your browser at `http://localhost:8501`.

### 4. Run the REST API (FastAPI)
To launch the backend API:
```bash
python -m uvicorn app.api:app --reload --port 8000
```
Interactive API docs will be available at: `http://localhost:8000/docs`.
- **POST `/chat`**: Accepts `{"message": "..."}`, returns full pipeline response and diagnostics.
- **POST `/analyze`**: Accepts `{"text": "..."}`, returns intermediate language, sentiment, and intent predictions.

---

## Known Limitations & Design Notes

1. **Granular Field Coverage in Benchmark Dataset (Bitext)**:
   - In the underlying `bitext/Bitext-customer-support-llm-chatbot-training-dataset`, the `ACCOUNT` category contains intents like `edit_account`, `switch_account`, and `recover_password`. However, specific fields (such as direct mentions of `"email address"`) are represented abstractly as `"personal information"`, `"account data"`, and generic platform navigation steps (`Account Settings`).
   - Rather than introducing synthetic or ad-hoc overrides, the pipeline relies on **Intent-Guided Retrieval** paired with strict **Sentiment-Conditioned Tone Prompting**. For neutral inquiries, the system provides standard account navigation procedures without unwarranted apologies or false escalations.

2. **Sentiment-Conditioned Prompting Architecture**:
   - Apologies and priority escalation notices are strictly constrained to genuine negative sentiment (`detected_sentiment == 'negative'`) or explicit complaints.
   - Neutral and positive inquiries receive direct, professional, and helpful customer support answers aligned with retail standards.