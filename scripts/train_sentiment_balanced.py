import os
import re
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

LABEL_MAPPING = {
    0: 0, # sadness -> negative
    1: 1, # joy -> positive
    2: 1, # love -> positive
    3: 0, # anger -> negative
    4: 0, # fear -> negative
    5: 2  # surprise -> neutral
}
TARGET_NAMES = ['negative', 'positive', 'neutral']

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\{\{[^}]+\}\}", "", text) # clean placeholders like {{Order Number}}
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", "", text)
    return text.strip()

class SimpleTokenizer:
    def __init__(self, max_vocab=15000):
        self.max_vocab = max_vocab
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        
    def fit(self, texts):
        word_counts = {}
        for text in texts:
            tokens = clean_text(text).split()
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1
                
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_words[:self.max_vocab - 2]:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
    def encode(self, text, max_len=64):
        tokens = clean_text(text).split()
        seq = [self.word2idx.get(t, self.word2idx["<UNK>"]) for t in tokens[:max_len]]
        if len(seq) < max_len:
            seq += [self.word2idx["<PAD>"]] * (max_len - len(seq))
        return seq

    def save(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({"word2idx": self.word2idx, "max_vocab": self.max_vocab}, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tok = cls(max_vocab=data["max_vocab"])
        tok.word2idx = data["word2idx"]
        tok.idx2word = {int(v): k for k, v in data["word2idx"].items()}
        return tok

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        seq = self.tokenizer.encode(self.texts[idx], self.max_len)
        return torch.tensor(seq, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

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
        pooled, _ = torch.max(out, dim=1) # Global max pooling
        x = self.dropout(self.relu(self.fc1(pooled)))
        logits = self.fc2(x)
        return logits

def main():
    print(">>> [Phase 3] Loading dair-ai/emotion and supplementing e-commerce domain data...")
    emotion_ds = load_dataset("dair-ai/emotion")
    bitext_ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")['train']
    
    # Extract neutral support questions and complaints from bitext
    neutral_instructions = [
        row['instruction'] for row in bitext_ds 
        if row['intent'] in ['track_order', 'delivery_options', 'check_invoice', 'get_invoice', 'create_account', 'recover_password']
    ]
    random.shuffle(neutral_instructions)
    neutral_instructions = neutral_instructions[:3000] # 3000 neutral customer queries
    
    complaint_instructions = [
        row['instruction'] for row in bitext_ds 
        if row['intent'] in ['complaint', 'payment_issue']
    ]
    random.shuffle(complaint_instructions)
    complaint_instructions = complaint_instructions[:1000] # 1000 frustrated queries
    
    # Train data
    train_texts = [x['text'] for x in emotion_ds['train']]
    train_labels = [LABEL_MAPPING[x['label']] for x in emotion_ds['train']]
    
    # Add domain supplements to train
    train_texts.extend(neutral_instructions[:2500])
    train_labels.extend([2] * 2500) # Neutral
    
    train_texts.extend(complaint_instructions[:800])
    train_labels.extend([0] * 800) # Negative
    
    # Validation data
    val_texts = [x['text'] for x in emotion_ds['validation']]
    val_labels = [LABEL_MAPPING[x['label']] for x in emotion_ds['validation']]
    val_texts.extend(neutral_instructions[2500:2750])
    val_labels.extend([2] * 250)
    val_texts.extend(complaint_instructions[800:900])
    val_labels.extend([0] * 100)
    
    # Test data
    test_texts = [x['text'] for x in emotion_ds['test']]
    test_labels = [LABEL_MAPPING[x['label']] for x in emotion_ds['test']]
    test_texts.extend(neutral_instructions[2750:])
    test_labels.extend([2] * len(neutral_instructions[2750:]))
    test_texts.extend(complaint_instructions[900:])
    test_labels.extend([0] * len(complaint_instructions[900:]))
    
    print(f"Dataset Counts (Enriched with Support Domain):")
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    
    # Tokenizer
    tokenizer = SimpleTokenizer(max_vocab=15000)
    tokenizer.fit(train_texts)
    print(f"Vocabulary size: {len(tokenizer.word2idx)}")
    
    train_ds = TextDataset(train_texts, train_labels, tokenizer)
    val_ds = TextDataset(val_texts, val_labels, tokenizer)
    test_ds = TextDataset(test_texts, test_labels, tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    
    # Balanced loss
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)
    weights = total_samples / (len(class_counts) * class_counts)
    class_weights = torch.tensor(weights, dtype=torch.float)
    print(f"Class counts: {class_counts}, Weights: {class_weights}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    
    model = BiLSTMSentiment(vocab_size=len(tokenizer.word2idx), embed_dim=128, hidden_dim=128, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
    
    epochs = 5
    best_val_f1 = 0.0
    
    print("\n--- Training Domain-Aware Bi-LSTM Sentiment Classifier ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += loss.item()
            
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(batch_y.cpu().numpy())
                
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        scheduler.step(val_f1)
        
        print(f"Epoch [{epoch}/{epochs}] - Loss: {total_loss/len(train_loader):.4f} - Val Acc: {val_acc:.4f} - Val Macro F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/sentiment_bilstm.pt")
            tokenizer.save("models/sentiment_tokenizer.json")
            
    # Evaluation on Test Set
    print("\n--- Final Test Set Evaluation ---")
    model.load_state_dict(torch.load("models/sentiment_bilstm.pt", map_location=device))
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_targets.extend(batch_y.cpu().numpy())
            
    test_acc = accuracy_score(test_targets, test_preds)
    test_f1 = f1_score(test_targets, test_preds, average='macro')
    print(f"Test Accuracy: {test_acc:.4f}, Test Macro F1: {test_f1:.4f}\n")
    print("Classification Report:\n", classification_report(test_targets, test_preds, target_names=TARGET_NAMES, digits=4))
    
    # Save Confusion Matrix
    cm = confusion_matrix(test_targets, test_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
    plt.title(f"Sentiment Classifier Confusion Matrix (Acc: {test_acc:.2%})", fontsize=13)
    plt.xlabel("Predicted Sentiment")
    plt.ylabel("True Sentiment")
    plt.tight_layout()
    plt.savefig("models/sentiment_cm.png", dpi=200)
    plt.close()
    print("Saved confusion matrix plot to models/sentiment_cm.png")
    
    # Realistic test queries
    test_cases = [
        ("Where is my order? I placed it last Monday.", "neutral"),
        ("I have been waiting for three weeks and no one answers, this is the worst service ever!", "negative"),
        ("Thank you so much for the swift refund, you have been super helpful!", "positive"),
        ("Why was my card charged twice for the same transaction?! This is ridiculous!", "negative"),
        ("Could you please explain how to reset my account credentials?", "neutral"),
        ("I am really happy with the item, it works great!", "positive"),
        ("How can I track my shipment to New York?", "neutral"),
        ("Cancel my order immediately, your company is completely fraudulent!", "negative")
    ]
    
    print("\n--- Domain-Specific Real Customer Support Inquiries ---")
    results = {}
    softmax = nn.Softmax(dim=1)
    for q, expected in test_cases:
        seq = torch.tensor([tokenizer.encode(q)], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(seq)
            probs = softmax(logits).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            pred_label = TARGET_NAMES[pred_idx]
            confidence = float(probs[pred_idx])
            prob_dict = {name: float(p) for name, p in zip(TARGET_NAMES, probs)}
            status = "PASS" if pred_label == expected else "FAIL"
            print(f"[{status}] Query: '{q}' -> Predicted: {pred_label} (Expected: {expected}, Conf: {confidence:.1%})")
            results[q] = {"predicted": pred_label, "expected": expected, "confidence": confidence, "probabilities": prob_dict}
            
    meta = {
        "model_architecture": "Domain-Augmented Bidirectional LSTM with Global Max-Pooling",
        "vocab_size": len(tokenizer.word2idx),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_f1),
        "target_names": TARGET_NAMES,
        "sample_tests": results
    }
    with open("models/sentiment_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("Saved metadata to models/sentiment_meta.json")

if __name__ == "__main__":
    main()
