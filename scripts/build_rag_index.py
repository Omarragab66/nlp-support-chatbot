import os
import re
import json
import faiss
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

def clean_text(text):
    text = re.sub(r"\{\{[^}]+\}\}", "", text)
    return " ".join(text.split())

def main():
    print(">>> [Phase 5] Loading Knowledge Base from bitext dataset...")
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")['train']
    
    # Extract distinct question-answer pairs
    df = pd.DataFrame(ds)
    print(f"Total raw pairs in dataset: {len(df):,}")
    
    # Group by intent/category to sample diverse knowledge chunks
    # To keep vector search fast, precise, and memory efficient,
    # we take up to 200 high-quality unique Q&A pairs per intent across all 27 intents
    kb_records = []
    seen_instructions = set()
    
    for _, row in df.iterrows():
        inst = clean_text(row['instruction'])
        resp = clean_text(row['response'])
        key = (row['intent'], resp)
        if key not in seen_instructions and len(inst) > 10 and len(resp) > 20:
            seen_instructions.add(key)
            kb_records.append({
                "instruction": inst,
                "response": resp,
                "category": row['category'],
                "intent": row['intent']
            })
            
    print(f"Curated Knowledge Base Chunks: {len(kb_records):,} high-quality Q&A pairs across all categories")
    
    # Embedding Model
    print("\nLoading sentence-transformers embedding model: 'all-MiniLM-L6-v2'...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Instructions to embed
    instructions = [item['instruction'] for item in kb_records]
    print(f"Generating embeddings for {len(instructions):,} support instructions...")
    embeddings = embedder.encode(instructions, show_progress_bar=True, batch_size=64, convert_to_numpy=True)
    
    # Normalize embeddings for cosine similarity with inner product index
    faiss.normalize_L2(embeddings)
    embedding_dim = embeddings.shape[1]
    print(f"Embeddings shape: {embeddings.shape} (Dimension: {embedding_dim})")
    
    # Build FAISS Index (IndexFlatIP for exact cosine similarity)
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)
    print(f"FAISS Index built with {index.ntotal:,} vectors.")
    
    # Save artifacts
    os.makedirs("models", exist_ok=True)
    faiss_index_path = "models/faiss_index.bin"
    faiss.write_index(index, faiss_index_path)
    print(f"Saved FAISS index to {faiss_index_path}")
    
    kb_chunks_path = "models/kb_chunks.json"
    with open(kb_chunks_path, "w", encoding="utf-8") as f:
        json.dump(kb_records, f, indent=2, ensure_ascii=False)
    print(f"Saved Knowledge Base chunks to {kb_chunks_path}")
    
    # Quick verification test
    test_query = "How can I return an item and get a refund?"
    q_emb = embedder.encode([test_query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, k=3)
    
    print(f"\n--- FAISS Search Verification for Query: '{test_query}' ---")
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        match = kb_records[idx]
        print(f"\nTop {rank} (Cosine Similarity: {score:.4f}):")
        print(f"  Matched Instruction: {match['instruction']}")
        print(f"  Category / Intent: {match['category']} / {match['intent']}")
        print(f"  Response Snippet: {match['response'][:120]}...")

if __name__ == "__main__":
    main()
