import sys, os
sys.path.insert(0, os.path.abspath("."))
import faiss
import numpy as np
from src.pipeline import CustomerSupportPipeline

bot = CustomerSupportPipeline()
q = "My package arrived damaged and I've been waiting a week for a response, I'm really upset!"

intent, conf = bot.detect_intent(q)
print(f"ML Intent Classifier: {intent} ({conf:.1%})")

INTENT_TO_KB_MAP = {
    "order_status": {"track_order", "delivery_options", "delivery_period"},
    "order_management": {"cancel_order", "change_order", "place_order", "change_shipping_address", "set_up_shipping_address"},
    "billing_and_refunds": {"check_invoice", "get_invoice", "check_payment_methods", "payment_issue", "check_refund_policy", "get_refund", "track_refund", "check_cancellation_fee"},
    "account_management": {"create_account", "edit_account", "delete_account", "switch_account", "recover_password", "registration_problems", "newsletter_subscription", "contact_customer_service"},
    "complaint": {"complaint", "review", "contact_human_agent"}
}

target_kb_intents = INTENT_TO_KB_MAP.get(intent, set())

# Find all indices in bot.kb_chunks belonging to target_kb_intents
target_indices = [i for i, c in enumerate(bot.kb_chunks) if c.get("intent") in target_kb_intents]
print(f"Total KB chunks for intent '{intent}': {len(target_indices)}")

# Extract embeddings of these target chunks from faiss index or reconstruct
# Since IndexFlatIP stores all raw vectors:
all_vectors = bot.faiss_index.reconstruct_n(0, bot.faiss_index.ntotal)
target_vectors = all_vectors[target_indices]

# Build temporary sub-index or compute dot product
sub_index = faiss.IndexFlatIP(target_vectors.shape[1])
sub_index.add(target_vectors)

q_emb = bot.embedder.encode([q], convert_to_numpy=True)
faiss.normalize_L2(q_emb)

scores, sub_indices = sub_index.search(q_emb, 5)

print("\n--- Intent-Filtered Retrieval Results ---")
for rank, (score, sub_idx) in enumerate(zip(scores[0], sub_indices[0]), 1):
    original_idx = target_indices[sub_idx]
    chunk = bot.kb_chunks[original_idx]
    print(f"Rank {rank} [Cosine Similarity: {score:.3f} | Intent: {chunk['intent']}]:")
    print(f"  Instruction: {chunk['instruction']}")
    print(f"  Response: {chunk['response'][:130]}...")
