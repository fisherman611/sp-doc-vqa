import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from tqdm.auto import tqdm
import numpy as np

from utils.helper import load_config

config = load_config('models/multimodal_rag/multimodal_classifier/clip/config.json')

DEVICE = config['device']['cuda'] if torch.cuda.is_available() else config['device']['cpu']
MODEL = config['model']
NUM_CLASSES = len(config['classes'])
MAX_LEN = config['max_len']

def train_model(model, train_loader, val_loader, label2id, epochs=5, lr=2e-5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    model.to(DEVICE)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            imgs = batch["pixel_values"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(ids, mask, imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Train loss: {total_loss / len(train_loader):.4f}")
        evaluate(model, val_loader, label2id)

@torch.no_grad()
def evaluate(model, val_loader, label2id, threshold=0.5):
    model.eval()
    all_preds, all_labels = [], []
    for batch in val_loader:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        imgs = batch["pixel_values"].to(DEVICE)
        lbls = batch["label"].cpu().numpy()  # shape [B, num_classes]

        logits = model(ids, mask, imgs)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > threshold).astype(int)

        all_preds.append(preds)
        all_labels.append(lbls)

    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_preds)

    # === Metrics ===
    subset_acc = (y_true == y_pred).all(axis=1).mean()  # exact match accuracy
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")

    # Micro accuracy (Jaccard-based)
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    micro_acc = intersection / union if union > 0 else 0.0

    print(f"SubsetAcc={subset_acc:.4f} | MicroAcc={micro_acc:.4f} | "
          f"MacroF1={macro_f1:.4f} | MicroF1={micro_f1:.4f}")

    return subset_acc, micro_acc, macro_f1, micro_f1

