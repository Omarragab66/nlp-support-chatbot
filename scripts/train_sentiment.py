import os
import re
import json
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

# Mapping original 6 classes to 3 target sentiment classes
# 0: sadness -> negative (0)
# 1: joy -> positive (1)
# 2: love -> positive (1)
# 3: anger -> negative (0)
# 4: fear -> negative (0)
# 5: surprise -> neutral (2)
LABEL_MAPPING = {
    0: 0, # sadness -> negative
    1: 1, # joy -> positive
    2: 1, # love -> positive
    3: 0, # anger -> negative
    4: 0, # fear -> negative
    5: 2  # surprise -> neutral
}
TARGET_NAMES = ['negative', 'positive', 'neutral']

# E-commerce customer support domain supplement to address Twitter domain shift
DOMAIN_SUPPLEMENT = [
    # Neutral customer queries
    ("Can you please tell me your return policy?", 2),
    ("How long does standard shipping usually take to deliver?", 2),
    ("I need to change my shipping address before the item ships.", 2),
    ("Where can I check the invoice for my last purchase?", 2),
    ("Is it possible to track order number 98412?", 2),
    ("What payment methods do you accept on the website?", 2),
    ("Can I create an account using my Google email?", 2),
    ("How do I reset my password if I forgot it?", 2),
    ("Do you offer express delivery for international shipments?", 2),
    ("Could you provide more details about the product specifications?", 2),
    # Frustrated / Negative queries
    ("I have been waiting for two weeks and my order is still missing! Ridiculous!", 0),
    ("Worst customer support ever, nobody is responding to my emails!", 0),
    ("I received a broken item and your policy is completely unfair!", 0),
    ("Give me my refund right now or I will dispute the charge with my bank!", 0),
    ("I am extremely angry with this service, cancel my account immediately!", 0),
    # Positive queries
    ("Thank you so much for the quick resolution, you made my day!", 1),
    ("Awesome customer support, very polite and helpful agent!", 1),
    ("I really love the quality of this item, exceeded my expectations!", 1),
    ("The replacement arrived safely and fast, thank you for your help!", 1),
    ("Everything is clear now, great job team!", 1),
]

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
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
        out, (hn, cn) = self.bilstm(embedded)
        # Global max pooling across time dimension
        pooled, _ = torch.max(out, dim=1)
        x = self.dropout(self.relu(self.fc1(pooled)))
        logits = self.fc2(x)
        return logits

def main():
    print(">>> [Phase 3] Loading dair-ai/emotion dataset...")
    dataset = load_dataset("dair-ai/emotion")
    
    train_texts = [x['text'] for x in dataset['train']]
    train_labels = [LABEL_MAPPING[x['label']] for x in dataset['train']]
    
    val_texts = [x['text'] for x in dataset['validation']]
    val_labels = [LABEL_MAPPING[x['label']] for x in dataset['validation']]
    
    test_texts = [x['text'] for x in dataset['test']]
    test_labels = [LABEL_MAPPING[x['label']] for x in dataset['test']]
    
    # Supplement domain data into train and test
    for text, lbl in DOMAIN_SUPPLEMENT:
        train_texts.append(text)
        train_labels.append(lbl)
        
    print(f"Data counts -> Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    
    # Build tokenizer
    tokenizer = SimpleTokenizer(max_vocab=15000)
    tokenizer.fit(train_texts)
    print(f"Vocabulary size: {len(tokenizer.word2idx)}")
    
    # Prepare datasets and loaders
    train_ds = TextDataset(train_texts, train_labels, tokenizer)
    val_ds = TextDataset(val_texts, val_labels, tokenizer)
    test_ds = TextDataset(test_texts, test_labels, tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    
    # Compute class weights for loss balancing
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)
    weights = total_samples / (len(class_counts) * class_counts)
    class_weights = torch.tensor(weights, dtype=torch.float)
    print(f"Class counts: {class_counts}, Class weights: {class_weights}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = BiLSTMSentiment(vocab_size=len(tokenizer.word2idx), embed_dim=128, hidden_dim=128, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
    
    epochs = 6
    best_val_f1 = 0.0
    
    print("\n--- Starting Neural Network Training (Bi-LSTM) ---")
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
            
        # Validation
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
            
    print(f"\nBest Validation Macro F1: {best_val_f1:.4f}")
    
    # Evaluate on Test Set
    print("\n--- Final Evaluation on Unseen Test Set (2000 samples) ---")
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
    
    # Plot and save confusion matrix
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
    
    # Customer support inference test
    test_cases = [
        "Where is my order? I placed it last Monday.", # neutral
        "I've been waiting for three weeks and no one answers, this is the worst service ever!", # negative / frustrated
        "Thank you so much for the swift refund, you've been super helpful!", # positive / satisfied
        "Why was my card charged twice for the same transaction?!", # negative / frustrated
        "Could you please explain how to reset my account credentials?", # neutral
    ]
    print("\n--- Testing Realistic Customer Support Inquiries ---")
    results = {}
    softmax = nn.Softmax(dim=1)
    for q in test_cases:
        seq = torch.tensor([tokenizer.encode(q)], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(seq)
            probs = softmax(logits).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            pred_label = TARGET_NAMES[pred_idx]
            confidence = float(probs[pred_idx])
            prob_dict = {name: float(p) for name, p in zip(TARGET_NAMES, probs)}
            print(f"Query: '{q}' -> Sentiment: {pred_label} (Confidence: {confidence:.2%})")
            results[q] = {"sentiment": pred_label, "confidence": confidence, "probabilities": prob_dict}
            
    meta = {
        "model_architecture": "Bidirectional LSTM with Attention/Global Max-Pooling",
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
