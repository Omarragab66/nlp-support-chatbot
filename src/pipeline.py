import os
import re
import json
import joblib
import faiss
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

# Load environment
load_dotenv()

# --- PyTorch Sentiment Model Definitions ---
class SimpleTokenizer:
    def __init__(self, max_vocab=15000):
        self.max_vocab = max_vocab
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        
    def encode(self, text, max_len=64):
        clean = re.sub(r"[^a-zA-Z0-9\s.,!?']", "", text.lower())
        tokens = clean.split()
        seq = [self.word2idx.get(t, self.word2idx["<UNK>"]) for t in tokens[:max_len]]
        if len(seq) < max_len:
            seq += [self.word2idx["<PAD>"]] * (max_len - len(seq))
        return seq

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tok = cls(max_vocab=data["max_vocab"])
        tok.word2idx = data["word2idx"]
        tok.idx2word = {int(v): k for k, v in data["word2idx"].items()}
        return tok

class BiLSTMSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=3, num_layers=2, dropout=0.3):
        super(BiLSTMSentiment, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        out, _ = self.bilstm(embedded)
        pooled, _ = torch.max(out, dim=1)
        x = self.dropout(self.relu(self.fc1(pooled)))
        return self.fc2(x)

class CustomerSupportPipeline:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sentiment_labels = ['negative', 'positive', 'neutral']
        
        print(">>> Initializing Unified Customer Support Pipeline...")
        
        # 1. Language Detector
        lang_model_path = os.path.join(models_dir, "language_detector.pkl")
        print(f"Loading Language Detector from {lang_model_path}...")
        self.lang_detector = joblib.load(lang_model_path)
        
        # 2. Sentiment Classifier
        sent_weights_path = os.path.join(models_dir, "sentiment_bilstm.pt")
        sent_tok_path = os.path.join(models_dir, "sentiment_tokenizer.json")
        print(f"Loading Sentiment BiLSTM from {sent_weights_path}...")
        self.sent_tokenizer = SimpleTokenizer.load(sent_tok_path)
        self.sent_model = BiLSTMSentiment(vocab_size=len(self.sent_tokenizer.word2idx)).to(self.device)
        self.sent_model.load_state_dict(torch.load(sent_weights_path, map_location=self.device))
        self.sent_model.eval()
        self.softmax = nn.Softmax(dim=1)
        
        # 3. Intent Classifier
        intent_model_path = os.path.join(models_dir, "intent_classifier.pkl")
        print(f"Loading Intent Classifier from {intent_model_path}...")
        self.intent_classifier = joblib.load(intent_model_path)
        
        # 4. RAG Components
        print("Loading Sentence Transformer ('all-MiniLM-L6-v2')...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        faiss_path = os.path.join(models_dir, "faiss_index.bin")
        kb_path = os.path.join(models_dir, "kb_chunks.json")
        print(f"Loading FAISS index from {faiss_path}...")
        self.faiss_index = faiss.read_index(faiss_path)
        
        with open(kb_path, "r", encoding="utf-8") as f:
            self.kb_chunks = json.load(f)
            
        # Groq Client
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_client = Groq(api_key=self.groq_api_key) if self.groq_api_key else None
        print("Pipeline initialized successfully!\n")

    def detect_language(self, text):
        pred_lang = self.lang_detector.predict([text])[0]
        probs = self.lang_detector.predict_proba([text])[0]
        conf = float(np.max(probs))
        return pred_lang, conf

    def detect_sentiment(self, text):
        seq = torch.tensor([self.sent_tokenizer.encode(text)], dtype=torch.long).to(self.device)
        with torch.no_grad():
            logits = self.sent_model(seq)
            probs = self.softmax(logits).cpu().numpy()[0]
            idx = int(np.argmax(probs))
            return self.sentiment_labels[idx], float(probs[idx]), {self.sentiment_labels[i]: float(probs[i]) for i in range(3)}

    def detect_intent(self, text):
        probs = self.intent_classifier.predict_proba([text])[0]
        max_idx = int(np.argmax(probs))
        pred_intent = self.intent_classifier.classes_[max_idx]
        conf = float(probs[max_idx])
        
        # Out-of-scope threshold
        if conf < 0.35:
            pred_intent = "out_of_scope"
            
        return pred_intent, conf

    def retrieve_chunks(self, query, top_k=3, min_similarity=0.35):
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        scores, indices = self.faiss_index.search(q_emb, top_k)
        
        retrieved = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= min_similarity:
                chunk = self.kb_chunks[idx]
                retrieved.append({
                    "score": float(score),
                    "instruction": chunk["instruction"],
                    "response": chunk["response"],
                    "category": chunk.get("category", ""),
                    "intent": chunk.get("intent", "")
                })
        return retrieved

    def generate_llm_answer(self, user_message, chunks, detected_sentiment):
        if not self.groq_client:
            return "I apologize, but the Groq API key is not configured. Please set GROQ_API_KEY in .env."
            
        context_str = "\n\n".join([f"Support Doc {i+1}: {c['response']}" for i, c in enumerate(chunks)])
        
        system_prompt = f"""You are a helpful, professional customer support assistant for an online retailer.
Answer the customer's question using ONLY the information in the retrieved support responses below.
If the customer sounds frustrated ({detected_sentiment}), acknowledge their frustration with a polite and sincere apology before answering.
If the retrieved context does not cover the question, say so honestly and offer to escalate to a human agent rather than guessing."""

        user_prompt = f"""Context (retrieved past support responses):
{context_str}

Customer question: "{user_message}" """

        try:
            resp = self.groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=250,
                temperature=0.2
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"Error communicating with LLM: {str(e)}"

    def process_message(self, user_message):
        user_message = user_message.strip()
        if not user_message:
            return {"response": "Please enter a question or message.", "status": "empty"}

        # Step 1: Language Detection
        lang, lang_conf = self.detect_language(user_message)

        # Step 2: Sentiment Detection
        sentiment, sent_conf, sent_probs = self.detect_sentiment(user_message)

        # Step 3: Intent Classification
        intent, intent_conf = self.detect_intent(user_message)

        # Step 4: Routing & Response Logic
        escalated = False
        grounded = False
        retrieved_chunks = []
        routing_action = "Standard Grounded RAG"

        # Routing Branch 1: Greetings & Small Talk (Zero retrieval needed)
        if intent == "greeting_smalltalk":
            routing_action = "Direct Smalltalk Response (No RAG)"
            if sentiment == "positive":
                final_response = "Hello! It's a pleasure to assist you today. How can I help you with your order or account?"
            else:
                final_response = "Hello! Welcome to our customer support. How can I assist you today?"

        # Routing Branch 2: Out of Scope
        elif intent == "out_of_scope":
            routing_action = "Out-of-Scope Fallback"
            final_response = "I am an e-commerce customer support assistant focused on retail orders, deliveries, refunds, and account queries. I'm unable to assist with topics outside this scope, but if you need help with our store, please let me know or I can connect you with a representative!"

        # Routing Branch 3: Formal Complaint / High Frustration
        elif intent == "complaint" or (sentiment == "negative" and intent == "billing_and_refunds"):
            routing_action = "Priority Escalation + Empathetic RAG"
            escalated = True
            chunks = self.retrieve_chunks(user_message, top_k=3)
            retrieved_chunks = chunks
            if chunks:
                rag_answer = self.generate_llm_answer(user_message, chunks, detected_sentiment=sentiment)
                final_response = f"I am truly sorry for the inconvenience and frustration you have experienced.\n\n{rag_answer}\n\n[System Notice]: Your inquiry has been flagged for priority human support. A senior customer care agent will review this shortly."
                grounded = True
            else:
                final_response = "I sincerely apologize for the frustration this has caused you. I have flagged your issue for urgent review by a human support manager who will assist you directly."

        # Routing Branch 4: Standard Store Inquiries (Order status, order management, billing, account)
        else:
            chunks = self.retrieve_chunks(user_message, top_k=3)
            retrieved_chunks = chunks
            if chunks:
                final_response = self.generate_llm_answer(user_message, chunks, detected_sentiment=sentiment)
                grounded = True
            else:
                final_response = "I apologize, but I couldn't find specific details for your question in our support records. Would you like me to connect you with a human representative for further assistance?"

        # If detected language is non-English, add a helpful note
        if lang != "en" and lang_conf > 0.85:
            final_response = f"[Note: Query detected in language '{lang}']: " + final_response

        return {
            "response": final_response,
            "detected_language": lang,
            "language_confidence": lang_conf,
            "detected_sentiment": sentiment,
            "sentiment_confidence": sent_conf,
            "sentiment_probabilities": sent_probs,
            "detected_intent": intent,
            "intent_confidence": intent_conf,
            "routing_action": routing_action,
            "priority_escalation": escalated,
            "grounded": grounded,
            "retrieved_chunks_count": len(retrieved_chunks),
            "retrieved_chunks": retrieved_chunks
        }
