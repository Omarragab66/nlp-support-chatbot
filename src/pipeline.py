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

LANGUAGE_NAMES = {
    "ar": "Arabic",
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "tr": "Turkish",
    "nl": "Dutch",
    "pt": "Portuguese",
    "sw": "Swahili",
    "th": "Thai",
    "vi": "Vietnamese",
    "pl": "Polish",
    "el": "Greek",
    "bg": "Bulgarian",
    "hi": "Hindi",
    "ur": "Urdu"
}

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

        # Mapping coarse intents to fine-grained KB chunk intents
        self.intent_to_kb_map = {
            "order_status": {"track_order", "delivery_options", "delivery_period"},
            "order_management": {"cancel_order", "change_order", "place_order", "change_shipping_address", "set_up_shipping_address"},
            "billing_and_refunds": {"check_invoice", "get_invoice", "check_payment_methods", "payment_issue", "check_refund_policy", "get_refund", "track_refund", "check_cancellation_fee"},
            "account_management": {"create_account", "edit_account", "delete_account", "switch_account", "recover_password", "registration_problems", "newsletter_subscription", "contact_customer_service"},
            "complaint": {"complaint", "review", "contact_human_agent"}
        }
        
        # Pre-index chunk indices by coarse intent for fast, efficient filtering
        self.intent_chunk_indices = {}
        for coarse_intent, sub_intents in self.intent_to_kb_map.items():
            self.intent_chunk_indices[coarse_intent] = [
                i for i, c in enumerate(self.kb_chunks) if c.get("intent") in sub_intents
            ]
            
        # Groq Client
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_client = Groq(api_key=self.groq_api_key) if self.groq_api_key else None
        print("Pipeline initialized successfully!\n")

    def detect_language(self, text):
        pred_lang = self.lang_detector.predict([text])[0]
        probs = self.lang_detector.predict_proba([text])[0]
        conf = float(np.max(probs))
        return pred_lang, conf

    def translate_to_english(self, text, source_lang="ar"):
        if not self.groq_client or not text:
            return text
        try:
            resp = self.groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": "You are a professional customer support translator. Translate the following customer message accurately into clear English. Return ONLY the direct English translation without any notes or explanations."},
                    {"role": "user", "content": text}
                ],
                max_tokens=400,
                temperature=0.1
            )
            tr = resp.choices[0].message.content.strip()
            return tr if tr else text
        except Exception as e:
            print(f"Translation warning: {e}")
            return text

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

    def retrieve_chunks(self, query, top_k=3, min_similarity=0.35, target_intent=None):
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        
        # 1. Intent-Guided Retrieval:
        # If a domain intent is predicted, search within that intent's sub-pool first
        if target_intent and target_intent in self.intent_chunk_indices:
            candidate_indices = self.intent_chunk_indices[target_intent]
            if candidate_indices:
                # Search candidate sub-pool via fast dot product
                all_vectors = self.faiss_index.reconstruct_n(0, self.faiss_index.ntotal)
                sub_vectors = all_vectors[candidate_indices]
                
                # Cosine similarities = dot product of normalized vectors
                sim_scores = np.dot(sub_vectors, q_emb.T).flatten()
                
                # Get top_k indices sorted descending
                top_sub_idx = np.argsort(-sim_scores)[:top_k]
                
                retrieved = []
                for s_idx in top_sub_idx:
                    score = float(sim_scores[s_idx])
                    if score >= min_similarity:
                        chunk = self.kb_chunks[candidate_indices[s_idx]]
                        retrieved.append({
                            "score": score,
                            "instruction": chunk["instruction"],
                            "response": chunk["response"],
                            "category": chunk.get("category", ""),
                            "intent": chunk.get("intent", "")
                        })
                
                # If high-confidence matches were found in the predicted intent partition, return them
                if retrieved:
                    return retrieved

        # 2. Global FAISS Fallback:
        scores, indices = self.faiss_index.search(q_emb, top_k)
        retrieved = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= min_similarity and idx >= 0:
                chunk = self.kb_chunks[idx]
                retrieved.append({
                    "score": float(score),
                    "instruction": chunk["instruction"],
                    "response": chunk["response"],
                    "category": chunk.get("category", ""),
                    "intent": chunk.get("intent", "")
                })
        return retrieved

    def generate_llm_answer(self, user_message, chunks, detected_sentiment, target_language="en"):
        if not self.groq_client:
            return "I apologize, but the Groq API key is not configured. Please set GROQ_API_KEY in .env."
            
        context_str = "\n\n".join([f"Support Doc {i+1}: {c['response']}" for i, c in enumerate(chunks)])
        
        lang_name = LANGUAGE_NAMES.get(target_language, "English")
        if target_language != "en":
            lang_instruction = f"""
LANGUAGE REQUIREMENT: The customer wrote their message in {lang_name}.
You MUST formulate your ENTIRE final response in natural, fluent, and polite {lang_name}.
Translate all procedures, steps, and policies accurately into {lang_name}."""
        else:
            lang_instruction = ""
            
        # Dynamic Tone & Behavior Instruction based on Sentiment
        if detected_sentiment == "negative":
            tone_instruction = """
TONE & EMOTIONAL CONDITIONING: The customer appears frustrated, upset, or experienced an issue.
Acknowledge their feelings with a sincere, polite, and empathetic apology before addressing their concern.
If the retrieved context does not cover the question, acknowledge the issue honestly and offer priority escalation to a human agent."""
        else:
            tone_instruction = """
TONE & BEHAVIOR: The customer is making a regular inquiry with a neutral/polite tone.
Maintain a warm, clear, professional, and helpful retail support tone.
Do NOT apologize or sound apologetic when there is no mistake, defect, or complaint."""

        grounding_instruction = """
STRICT GROUNDING & ANTI-HALLUCINATION RULES:
1. Answer the customer's question using ONLY the facts and procedures present in the retrieved support responses.
2. Do NOT invent specific procedural details, UI element names, confirmation mechanisms (e.g. "a confirmation email will be sent"), or numbered walkthrough steps that are NOT explicitly present in the retrieved context.
3. When the retrieved context describes a topic in general or abstract terms (e.g. "access your account settings to update your information"), your answer MUST stay at that same level of generality. Do not fabricate a multi-step tutorial to make the answer look more detailed than the source material supports.
4. You may use clear formatting, polite phrasing, or bullet points for readability, provided every factual instruction traces directly to the retrieved documents."""

        system_prompt = f"""You are a helpful, professional customer support assistant for an online retailer.
Answer the customer's question using the information in the retrieved support responses below.{lang_instruction}
{tone_instruction}
{grounding_instruction}"""

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
                max_tokens=600,
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

        # Step 1.5: Multilingual Query Normalization
        # If query is in a non-English language (e.g. Arabic), translate to English
        # so English-trained intent classifier, sentiment classifier, and FAISS vector index work accurately.
        translated_query = None
        if lang != "en" and lang_conf >= 0.40:
            translated_query = self.translate_to_english(user_message, source_lang=lang)
            analysis_query = translated_query
        else:
            analysis_query = user_message

        # Step 2: Sentiment Detection on normalized text
        sentiment, sent_conf, sent_probs = self.detect_sentiment(analysis_query)

        # Step 3: Intent Classification on normalized text
        intent, intent_conf = self.detect_intent(analysis_query)

        # Step 4: Routing & Response Logic
        escalated = False
        grounded = False
        retrieved_chunks = []
        routing_action = "Standard Grounded RAG"

        # Routing Branch 1: Greetings & Small Talk (Zero retrieval needed)
        if intent == "greeting_smalltalk":
            routing_action = "Direct Smalltalk Response (No RAG)"
            if lang == "ar":
                final_response = "أهلاً بك! مرحباً بك في خدمة عملاء المتجر. كيف يمكنني مساعدتك اليوم بخصوص طلباتك، التوصيل، استرجاع المبالغ، أو إدارة حسابك؟"
            elif lang != "en":
                final_response = self.generate_llm_answer(
                    user_message,
                    chunks=[{"response": "Welcome to our store support. How can I assist you with your order or account today?"}],
                    detected_sentiment="positive",
                    target_language=lang
                )
            elif sentiment == "positive":
                final_response = "Hello! It's a pleasure to assist you today. How can I help you with your order or account?"
            else:
                final_response = "Hello! Welcome to our customer support. How can I assist you today?"

        # Routing Branch 2: Explicit Out of Scope
        elif intent == "out_of_scope":
            routing_action = "Out-of-Scope Fallback"
            if lang == "ar":
                final_response = "أنا مساعد آلي متخصص في خدمة عملاء المتجر (الطلبات، التوصيل، الفواتير، واسترجاع المنتجات). لا أستطيع المساعدة في مواضيع خارج هذا النطاق، ولكن إذا كان لديك أي استفسار يخص المتجر يسعدني جداً مساعدتك أو تحويلك لممثل خدمة العملاء!"
            elif lang != "en":
                final_response = self.generate_llm_answer(
                    user_message,
                    chunks=[{"response": "I am an e-commerce customer support assistant focused on retail orders, deliveries, refunds, and account queries. I'm unable to assist with topics outside this scope."}],
                    detected_sentiment="neutral",
                    target_language=lang
                )
            else:
                final_response = "I am an e-commerce customer support assistant focused on retail orders, deliveries, refunds, and account queries. I'm unable to assist with topics outside this scope, but if you need help with our store, please let me know or I can connect you with a representative!"

        # Routing Branch 3: Formal Complaint / Frustrated Customer (Escalation + Empathetic RAG)
        elif intent == "complaint" or sentiment == "negative":
            chunks = self.retrieve_chunks(analysis_query, top_k=3, target_intent=intent)
            retrieved_chunks = chunks
            
            # Grounding check: If no relevant store chunks matched at all, and it's not an explicit store complaint intent,
            # this is an out-of-scope / non-store query (e.g. general trivia/science triggering spurious features).
            if not chunks and intent != "complaint":
                routing_action = "Out-of-Scope Fallback"
                intent = "out_of_scope"
                intent_conf = max(intent_conf, 0.90)
                if lang == "ar":
                    final_response = "أنا مساعد آلي متخصص في خدمة عملاء المتجر (الطلبات، التوصيل، الفواتير، واسترجاع المنتجات). لا أستطيع المساعدة في مواضيع خارج هذا النطاق، ولكن إذا كان لديك أي استفسار يخص المتجر يسعدني جداً مساعدتك!"
                elif lang != "en":
                    final_response = self.generate_llm_answer(
                        user_message,
                        chunks=[{"response": "I am an e-commerce customer support assistant focused on retail orders, deliveries, refunds, and account queries. I'm unable to assist with topics outside this scope."}],
                        detected_sentiment="neutral",
                        target_language=lang
                    )
                else:
                    final_response = "I am an e-commerce customer support assistant focused on retail orders, deliveries, refunds, and account queries. I'm unable to assist with topics outside this scope, but if you need help with our store, please let me know!"
            elif chunks:
                routing_action = "Priority Escalation + Empathetic RAG"
                escalated = True
                rag_answer = self.generate_llm_answer(user_message, chunks, detected_sentiment=sentiment, target_language=lang)
                if lang == "ar":
                    final_response = f"{rag_answer}\n\n[إشعار النظام]: تم تصعيد تذكرتك بعناية إلى فريق الدعم البشري للمتابعة الفورية معك."
                else:
                    final_response = f"I am truly sorry for the inconvenience and frustration you have experienced.\n\n{rag_answer}\n\n[System Notice]: Your inquiry has been flagged for priority human support. A senior customer care agent will review this shortly."
                grounded = True
            else:
                routing_action = "Priority Escalation + Empathetic RAG"
                escalated = True
                if lang == "ar":
                    final_response = "أعتذر بشدة عن أي إزعاج واجهته. لقد قمت بتحويل طلبك لمشرف خدمة العملاء للتواصل معك وحل المشكلة فوراً."
                else:
                    final_response = "I sincerely apologize for the frustration this has caused you. I have flagged your issue for urgent review by a human support manager who will assist you directly."

        # Routing Branch 4: Standard Store Inquiries (Order status, order management, billing, account)
        else:
            chunks = self.retrieve_chunks(analysis_query, top_k=3, target_intent=intent)
            retrieved_chunks = chunks
            if chunks:
                final_response = self.generate_llm_answer(user_message, chunks, detected_sentiment=sentiment, target_language=lang)
                grounded = True
            else:
                # If zero store knowledge matched for a standard inquiry, treat gracefully as out-of-scope/unrecognized
                routing_action = "Out-of-Scope Fallback"
                if lang == "ar":
                    final_response = "أنا مساعد آلي متخصص في خدمة عملاء المتجر (الطلبات، الشحن، الفواتير، الحسابات). هذا الاستفسار خارج نطاق خدمات المتجر المتاحة، ولكن يسعدني دائماً مساعدتك في أي شيء يخص مشترياتك وطلباتك!"
                else:
                    final_response = "I am an e-commerce customer support assistant specialized in orders, shipping, billing, and account services. This request appears to be outside our store support scope, but I'm happy to help with any inquiries regarding our products or your orders!"

        return {
            "response": final_response,
            "detected_language": lang,
            "language_confidence": lang_conf,
            "translated_query": translated_query,
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
