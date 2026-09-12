import os
import re
import json
import joblib
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Set seed
random.seed(42)
np.random.seed(42)

# 27 intents mapping to 6 core categories
INTENT_MAPPING = {
    # order_status
    "track_order": "order_status",
    "delivery_options": "order_status",
    "delivery_period": "order_status",
    
    # order_management
    "cancel_order": "order_management",
    "change_order": "order_management",
    "place_order": "order_management",
    "change_shipping_address": "order_management",
    "set_up_shipping_address": "order_management",
    
    # billing_and_refunds
    "check_invoice": "billing_and_refunds",
    "get_invoice": "billing_and_refunds",
    "check_payment_methods": "billing_and_refunds",
    "payment_issue": "billing_and_refunds",
    "check_refund_policy": "billing_and_refunds",
    "get_refund": "billing_and_refunds",
    "track_refund": "billing_and_refunds",
    "check_cancellation_fee": "billing_and_refunds",
    
    # account_management
    "create_account": "account_management",
    "edit_account": "account_management",
    "delete_account": "account_management",
    "switch_account": "account_management",
    "recover_password": "account_management",
    "registration_problems": "account_management",
    "newsletter_subscription": "account_management",
    
    # complaint
    "complaint": "complaint",
    "review": "complaint",
    
    # contact / support routing
    "contact_customer_service": "account_management",
    "contact_human_agent": "complaint"
}

# Add Smalltalk examples (greeting, goodbye, gratitude)
SMALLTALK_EXAMPLES = [
    "hello", "hi there", "good morning", "good evening", "hey", "howdy",
    "how are you doing today?", "is anyone available?", "hello support team",
    "thank you so much", "thanks for your help", "appreciate your assistance",
    "that was very helpful thank you", "many thanks", "thanks a lot",
    "goodbye", "bye bye", "have a great day", "see you later", "talk to you soon",
    "ok thanks bye", "good night", "catch you later"
] * 40 # 920 samples

# Add Out-of-Scope examples
OUT_OF_SCOPE_EXAMPLES = [
    "what is the capital of france?", "who won the world cup in 2022?",
    "tell me a funny joke", "write a python code to reverse a string",
    "what is the weather like in tokyo today?", "can you solve 2x + 5 = 15?",
    "recommend a good science fiction movie", "how to bake a chocolate cake?",
    "what time is it in new york?", "tell me about quantum computing",
    "who is the ceo of google?", "what is the meaning of life?",
    "translate this sentence to spanish", "play my favorite music playlist"
] * 65 # 910 samples

def clean_instruction(text):
    text = text.lower()
    text = re.sub(r"\{\{[^}]+\}\}", " ", text) # remove {{Order Number}}, etc.
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", " ", text)
    return " ".join(text.split())

def main():
    print(">>> [Phase 4] Loading bitext/Bitext-customer-support dataset...")
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")['train']
    
    texts = []
    labels = []
    
    for row in ds:
        inst = clean_instruction(row['instruction'])
        raw_intent = row['intent']
        condensed_intent = INTENT_MAPPING.get(raw_intent, "out_of_scope")
        texts.append(inst)
        labels.append(condensed_intent)
        
    # Add Smalltalk & Out-of-Scope data
    for item in SMALLTALK_EXAMPLES:
        texts.append(clean_instruction(item))
        labels.append("greeting_smalltalk")
        
    for item in OUT_OF_SCOPE_EXAMPLES:
        texts.append(clean_instruction(item))
        labels.append("out_of_scope")
        
    df = pd.DataFrame({"text": texts, "intent": labels})
    print(f"Total labeled samples: {len(df):,}")
    print("\nCondensed Intent Distribution:")
    print(df['intent'].value_counts())
    
    # Train / Test split (80 / 20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['intent'], test_size=0.20, random_state=42, stratify=df['intent']
    )
    
    print(f"\nTrain size: {len(X_train):,}, Test size: {len(X_test):,}")
    
    # ML Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_features=25000
        )),
        ('clf', LogisticRegression(
            C=3.0,
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ))
    ])
    
    print("\nTraining Logistic Regression Intent Classifier...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    test_preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, test_preds)
    macro_f1 = f1_score(y_test, test_preds, average='macro')
    print(f"\nTest Accuracy: {acc:.4%}")
    print(f"Test Macro F1: {macro_f1:.4f}\n")
    
    unique_intents = sorted(df['intent'].unique().tolist())
    report = classification_report(y_test, test_preds, target_names=unique_intents, digits=4)
    print("Classification Report:\n", report)
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, test_preds, labels=unique_intents)
    os.makedirs("models", exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=unique_intents, yticklabels=unique_intents)
    plt.title(f"Intent Classifier Confusion Matrix (Acc: {acc:.2%})", fontsize=13)
    plt.xlabel("Predicted Intent")
    plt.ylabel("True Intent")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    cm_path = "models/intent_cm.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_path}")
    
    # Save Pipeline
    model_path = "models/intent_classifier.pkl"
    joblib.dump(pipeline, model_path)
    print(f"Saved trained intent pipeline to {model_path}")
    
    # Routing test cases
    test_queries = [
        ("Hi, good afternoon!", "greeting_smalltalk", "Direct Greeting Reply (No RAG needed)"),
        ("Where is my package and how long does standard shipping take?", "order_status", "Retrieve KB & generate status answer (RAG)"),
        ("I need to cancel order #48291 please", "order_management", "Retrieve cancel policy / order management path"),
        ("Can I get a full refund for my damaged item?", "billing_and_refunds", "Retrieve refund policy & process steps (RAG)"),
        ("I forgot my password, how can I recover my account?", "account_management", "Retrieve account recovery guide (RAG)"),
        ("Your service is awful, I want to file a formal complaint!", "complaint", "Empathetic apology + Human Escalation Flag"),
        ("What is the distance between the Earth and the Moon?", "out_of_scope", "Polite Out-of-Scope response / Offer human agent")
    ]
    
    print("\n--- Routing Decision Verification ---")
    results = {}
    for q, expected, expected_action in test_queries:
        probs = pipeline.predict_proba([q])[0]
        max_prob = float(np.max(probs))
        pred_intent = pipeline.classes_[np.argmax(probs)]
        
        # Out of scope threshold check
        CONFIDENCE_THRESHOLD = 0.35
        if max_prob < CONFIDENCE_THRESHOLD:
            routed_intent = "out_of_scope"
        else:
            routed_intent = pred_intent
            
        prob_dict = {cls: float(p) for cls, p in zip(pipeline.classes_, probs)}
        status = "PASS" if routed_intent == expected else "FAIL"
        print(f"[{status}] Query: '{q}' -> Intent: {routed_intent} (Conf: {max_prob:.1%}, Expected: {expected})")
        results[q] = {
            "predicted_intent": routed_intent,
            "expected_intent": expected,
            "confidence": max_prob,
            "routing_action": expected_action,
            "probabilities": prob_dict
        }
        
    meta = {
        "model": "TF-IDF (1-2 ngrams) + Balanced Logistic Regression",
        "test_accuracy": float(acc),
        "test_macro_f1": float(macro_f1),
        "classes": list(pipeline.classes_),
        "sample_tests": results
    }
    with open("models/intent_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("Saved metadata to models/intent_meta.json")

if __name__ == "__main__":
    main()
