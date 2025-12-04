import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score, hamming_loss, jaccard_score
from tqdm.auto import tqdm
import numpy as np

from utils.helper import load_config, label_recall_macro

config = load_config('models/multimodal_rag/multimodal_classifier/clip/config.json')

DEVICE = config['device']['cuda'] if torch.cuda.is_available() else config['device']['cpu']
MODEL = config['model']
NUM_CLASSES = len(config['classes'])
MAX_LEN = config['max_len']

def train_model(model, train_loader, val_loader, label2id, epochs=5, lr=2e-5, save_dir="checkpoints"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    model.to(DEVICE)
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pt")
    
    # Track best metrics
    best_jaccard = 0.0
    best_epoch = 0

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

        avg_train_loss = total_loss / len(train_loader)
        print(f"Train loss: {avg_train_loss:.4f}")
        
        # Evaluate and get metrics
        macro_label_recall, jaccard = evaluate(model, val_loader, label2id)
        
        # Save best model based on Jaccard score
        if jaccard > best_jaccard:
            best_jaccard = jaccard
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_jaccard': best_jaccard,
                'macro_label_recall': macro_label_recall
            }, best_model_path)
            print(f"✓ Best model saved at epoch {epoch+1} with Jaccard: {jaccard:.4f}")
    
    print(f"\nTraining completed! Best model from epoch {best_epoch} with Jaccard: {best_jaccard:.4f}")
    print(f"Best model saved at: {best_model_path}")
    
    return best_model_path

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
    macro_label_recall = label_recall_macro(y_true, y_pred)
    jaccard = jaccard_score(y_true, y_pred, average="samples", zero_division=0)

    print(f"Macro Label Recall={macro_label_recall:.4f} | Jaccard={jaccard:.4f}")

    return macro_label_recall, jaccard