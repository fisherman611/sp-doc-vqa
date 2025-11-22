import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
from sklearn.metrics import jaccard_score

from utils.helper import label_recall_macro

ALL_LABELS = [
    "handwritten",
    "form",
    "layout",
    "table/list",
    "others",
    "free_text",
    "Image/Photo",
    "figure/diagram",
    "Yes/No"
]
label2id = {lbl: i for i, lbl in enumerate(ALL_LABELS)}

def to_multi_hot(labels, label2id):
    vec = np.zeros(len(label2id), dtype=int)
    for lbl in labels:
        if lbl in label2id:
            vec[label2id[lbl]] = 1
    return vec

RESULT_PATH = "models/multimodal_rag/multimodal_classifier/gemini/results/zeroshot_results.json"

with open(RESULT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

Y_true = []
Y_pred = []

for item in data:
    gt = item.get("ground_truth", [])
    pred = item.get("predicted_labels", [])

    Y_true.append(to_multi_hot(gt, label2id))
    Y_pred.append(to_multi_hot(pred, label2id))

Y_true = np.vstack(Y_true)
Y_pred = np.vstack(Y_pred)


jaccard = jaccard_score(Y_true, Y_pred, average="samples", zero_division=0)
labelrec = label_recall_macro(Y_true, Y_pred)

print("===== GEMINI MULTILABEL CLASSIFIER METRICS =====")
print(f"Jaccard score      : {jaccard:.4f}")
print(f"Label Recall (macro): {labelrec:.4f}")
print("=================================================")
