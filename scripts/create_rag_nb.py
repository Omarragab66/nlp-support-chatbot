import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# Module 4: Q&A RAG Pipeline (محرك استرجاع وتوليد الإجابات)
**NLP Final Task 2026 - RAG-Based E-commerce Customer Support Chatbot**

## Overview
In this module, we build an end-to-end Retrieval-Augmented Generation (RAG) system grounded in real customer support policies and answers.
The RAG pipeline operates through three synchronized components:
1. **Dense Vector Retrieval**: Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to convert queries and knowledge chunks into semantic embeddings.
2. **Local Vector Database**: Built with **FAISS (Facebook AI Similarity Search)** using cosine similarity for ultra-fast, local indexing.
3. **Conditioned LLM Generation**: Powered by **Groq API** (`openai/gpt-oss-120b`), governed by dynamic system prompts that adjust tone (apology/empathy) according to the detected sentiment, while strictly constraining answers to retrieved context.

---
### Grounding & Hallucination Prevention
The model is explicitly instructed:
> *"Answer the customer's question using ONLY the information in the retrieved support responses below. If the customer sounds frustrated, acknowledge that before answering. If the retrieved context does not cover the question, say so honestly and offer to escalate to a human agent rather than guessing."*
"""))

# Cell 1: Imports
cells.append(nbf.v4.new_code_cell("""import os
import re
import json
import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv("../.env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(f"Groq API Key Configured: {bool(GROQ_API_KEY)}")
"""))

# Cell 2: Knowledge Base Loading
cells.append(nbf.v4.new_markdown_cell("""## 1. Load Knowledge Base & Inspect Q&A Chunks"""))

cells.append(nbf.v4.new_code_cell("""# Load pre-built knowledge base chunks
kb_path = "../models/kb_chunks.json"
if os.path.exists(kb_path):
    with open(kb_path, "r", encoding="utf-8") as f:
        kb_chunks = json.load(f)
    print(f"Loaded {len(kb_chunks):,} curated Q&A pairs from {kb_path}")
else:
    print("Loading from HuggingFace dataset...")
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")['train']
    kb_chunks = [{"instruction": r['instruction'], "response": r['response'], "category": r['category'], "intent": r['intent']} for r in ds]

pd.DataFrame(kb_chunks).head(5)
"""))

# Cell 3: Embeddings & FAISS Index
cells.append(nbf.v4.new_markdown_cell("""## 2. Load FAISS Vector Store & Embedding Model"""))

cells.append(nbf.v4.new_code_cell("""embedder = SentenceTransformer("all-MiniLM-L6-v2")

index_path = "../models/faiss_index.bin"
if os.path.exists(index_path):
    index = faiss.read_index(index_path)
    print(f"Loaded FAISS Index with {index.ntotal:,} vectors.")
else:
    print("Building FAISS index in memory...")
    instructions = [c['instruction'] for c in kb_chunks]
    embeddings = embedder.encode(instructions, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
"""))

# Cell 4: Retrieval Function
cells.append(nbf.v4.new_markdown_cell("""## 3. Semantic Retrieval Function"""))

cells.append(nbf.v4.new_code_cell("""def retrieve_support_chunks(query, top_k=3, min_similarity=0.35):
    query_vector = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vector)
    scores, indices = index.search(query_vector, top_k)
    
    retrieved = []
    for score, idx in zip(scores[0], indices[0]):
        if score >= min_similarity:
            chunk = kb_chunks[idx]
            retrieved.append({
                "score": float(score),
                "instruction": chunk["instruction"],
                "response": chunk["response"],
                "category": chunk.get("category", ""),
                "intent": chunk.get("intent", "")
            })
    return retrieved

# Test Retrieval
sample_q = "How can I return an order and request a refund?"
results = retrieve_support_chunks(sample_q, top_k=3)
print(f"Query: '{sample_q}'\\nRetrieved {len(results)} chunks:")
for i, res in enumerate(results, 1):
    print(f"\\n--- Chunk {i} (Similarity: {res['score']:.3f}, Intent: {res['intent']}) ---")
    print(f"Response: {res['response'][:180]}...")
"""))

# Cell 5: Prompt Template & LLM Generation
cells.append(nbf.v4.new_markdown_cell("""## 4. Sentiment-Conditioned Prompt Template & Groq Generation"""))

cells.append(nbf.v4.new_code_cell("""def build_rag_prompt(user_message, retrieved_chunks, detected_sentiment="neutral"):
    if retrieved_chunks:
        context_str = "\\n\\n".join([
            f"[Support Document {i+1}]: {c['response']}" 
            for i, c in enumerate(retrieved_chunks)
        ])
    else:
        context_str = "NO_RELEVANT_CONTEXT_FOUND"
        
    system_prompt = (
        "You are a helpful, professional customer support assistant for an online retailer.\\n"
        "Answer the customer's question using ONLY the information in the retrieved support responses below.\\n"
        f"If the customer sounds frustrated ({detected_sentiment}), acknowledge that with an empathetic, sincere apology before answering.\\n"
        "If the retrieved context does not cover the question, say so honestly and offer to escalate to a human agent rather than guessing."
    )

    user_prompt = f"Context (retrieved past support responses):\\n{context_str}\\n\\nCustomer question: \\\"{user_message}\\\""

    return system_prompt, user_prompt

def generate_rag_response(user_message, detected_sentiment="neutral", top_k=3):
    chunks = retrieve_support_chunks(user_message, top_k=top_k)
    system_prompt, user_prompt = build_rag_prompt(user_message, chunks, detected_sentiment)
    
    # If no relevant chunks found and not greeting
    if not chunks:
        return {
            "answer": "I apologize, but I couldn't find specific information regarding your request in our store policy. Would you like me to connect you with a human customer support specialist?",
            "grounded": False,
            "retrieved_chunks": []
        }
        
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=250
    )
    
    return {
        "answer": response.choices[0].message.content.strip(),
        "grounded": True,
        "retrieved_chunks": chunks
    }

print("RAG pipeline functions defined successfully!")
"""))

# Cell 6: End-to-End Pipeline Evaluation
cells.append(nbf.v4.new_markdown_cell("""## 5. End-to-End Test Scenarios

We evaluate the full RAG pipeline across several realistic retail scenarios:
1. **Frustrated customer with damaged item**: Requires empathetic tone + grounded refund policy.
2. **Neutral order tracking**: Standard direct information.
3. **Out-of-scope question**: Polite decline without hallucination.
"""))

cells.append(nbf.v4.new_code_cell(r"""test_cases = [
    {
        "query": "I received my order yesterday and it was completely broken! This is terrible, I demand a full refund right now!",
        "sentiment": "negative"
    },
    {
        "query": "How can I track the delivery status of my package?",
        "sentiment": "neutral"
    },
    {
        "query": "Can you explain quantum computing and black holes?",
        "sentiment": "neutral"
    }
]

for tc in test_cases:
    print()
    print("="*55)
    print(f"Customer: '{tc['query']}'")
    print(f"Detected Sentiment: {tc['sentiment']}")
    print("="*55)
    
    result = generate_rag_response(tc['query'], detected_sentiment=tc['sentiment'])
    print()
    print(f"Assistant Response:\n{result['answer']}")
    print(f"Grounded in KB: {result['grounded']} (Chunks Retrieved: {len(result['retrieved_chunks'])})")
"""))

nb.cells = cells

with open("notebooks/04_rag_pipeline.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Created notebooks/04_rag_pipeline.ipynb successfully!")
