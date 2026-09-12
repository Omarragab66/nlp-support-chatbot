import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    print(">>> [Phase 2] Loading papluca/language-identification dataset...")
    raw_dataset = load_dataset("papluca/language-identification")
    
    train_df = pd.DataFrame(raw_dataset['train'])
    val_df = pd.DataFrame(raw_dataset['validation'])
    test_df = pd.DataFrame(raw_dataset['test'])
    
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    languages = sorted(train_df['labels'].unique().tolist())
    print(f"Unique languages ({len(languages)}): {languages}")

    X_train, y_train = train_df['text'], train_df['labels']
    X_val, y_val = val_df['text'], val_df['labels']
    X_test, y_test = test_df['text'], test_df['labels']

    print("\n--- 1. Baseline: Word-level TF-IDF + MultinomialNB ---")
    baseline_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 1), max_features=25000)),
        ('nb', MultinomialNB())
    ])
    baseline_pipe.fit(X_train, y_train)
    baseline_val_preds = baseline_pipe.predict(X_val)
    baseline_val_acc = accuracy_score(y_val, baseline_val_preds)
    print(f"Baseline (Word TF-IDF + NB) Validation Accuracy: {baseline_val_acc:.4f}")

    print("\n--- 2. Enhancement 1: Subword Character n-grams (Char_wb 2-4) + MultinomialNB ---")
    char_nb_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=40000, sublinear_tf=True)),
        ('nb', MultinomialNB(alpha=0.1))
    ])
    char_nb_pipe.fit(X_train, y_train)
    char_nb_val_preds = char_nb_pipe.predict(X_val)
    char_nb_val_acc = accuracy_score(y_val, char_nb_val_preds)
    print(f"Enhancement 1 (Char TF-IDF + NB) Validation Accuracy: {char_nb_val_acc:.4f}")

    print("\n--- 3. Enhancement 2: Subword Char n-grams + SGDClassifier (Linear SVM / Modified Huber) ---")
    char_sgd_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), max_features=50000, sublinear_tf=True)),
        ('clf', SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-5, max_iter=30, random_state=42, n_jobs=-1))
    ])
    char_sgd_pipe.fit(X_train, y_train)
    char_sgd_val_preds = char_sgd_pipe.predict(X_val)
    char_sgd_val_acc = accuracy_score(y_val, char_sgd_val_preds)
    print(f"Enhancement 2 (Char TF-IDF + Linear SGD) Validation Accuracy: {char_sgd_val_acc:.4f}")

    # Select best model
    best_pipe = char_sgd_pipe if char_sgd_val_acc >= char_nb_val_acc else char_nb_pipe
    best_name = "Subword Char TF-IDF + Linear SGD" if best_pipe == char_sgd_pipe else "Subword Char TF-IDF + Naive Bayes"
    print(f"\nBest Model Selected: {best_name}")

    print("\n--- Final Evaluation on Unseen Test Set (10,000 samples) ---")
    test_preds = best_pipe.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    
    report = classification_report(y_test, test_preds, digits=4)
    print("\nClassification Report:\n", report)

    # Confusion matrix
    cm = confusion_matrix(y_test, test_preds, labels=languages)
    os.makedirs("models", exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=languages, yticklabels=languages)
    plt.title(f"Language Detection Confusion Matrix (Acc: {test_acc:.2%})", fontsize=14)
    plt.xlabel("Predicted Language")
    plt.ylabel("True Language")
    plt.tight_layout()
    cm_path = "models/lang_cm.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_path}")

    # Save model and artifacts
    model_path = "models/language_detector.pkl"
    joblib.dump(best_pipe, model_path)
    print(f"Saved best model pipeline to {model_path}")

    metadata = {
        "best_model": best_name,
        "test_accuracy": float(test_acc),
        "languages": languages,
        "char_ngram_range": [2, 5],
        "max_features": 50000,
        "sample_tests": {}
    }

    # Test realistic customer support queries
    sample_queries = [
        ("Where is my order #55412?", "en"),
        ("أين طلبي وكيف يمكنني استرجاع المبلغ؟", "ar"),
        ("Où est mon colis et comment puis-je être remboursé?", "fr"),
        ("Wo ist meine Bestellung? Ich möchte sie stornieren.", "de"),
        ("¿Dónde está mi paquete? Quiero cancelar mi pedido.", "es"),
        ("Dov'è il mio pacco? Vorrei fare un reso.", "it"),
        ("Где мой заказ и как оформить возврат?", "ru")
    ]
    print("\n--- Testing Realistic Customer Support Queries ---")
    for query, expected in sample_queries:
        pred = best_pipe.predict([query])[0]
        # Get probability if available
        probs = best_pipe.predict_proba([query])[0]
        confidence = float(np.max(probs))
        status = "PASS" if pred == expected else "FAIL"
        print(f"[{status}] Query: '{query}' -> Predicted: {pred} (Confidence: {confidence:.2%}, Expected: {expected})".encode('ascii', 'replace').decode())
        metadata["sample_tests"][query] = {"predicted": pred, "confidence": confidence, "expected": expected}

    with open("models/language_detector_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print("Saved metadata to models/language_detector_meta.json")

if __name__ == "__main__":
    main()
