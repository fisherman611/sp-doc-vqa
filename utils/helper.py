import json
import torch
import os
import numpy as np

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_best_model(model, checkpoint_path="checkpoints/best_model.pt", device="cuda"):
    """
    Load the best model checkpoint.
    
    Args:
        model: The model architecture to load weights into
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        model: Model with loaded weights
        checkpoint: Dictionary containing metrics and other info
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Loaded best model from epoch {checkpoint['epoch']}")
    print(f"Metrics: Jaccard={checkpoint['best_jaccard']:.4f}, "
          f"Macro Label Recall={checkpoint['macro_label_recall']:.4f}")
    
    return model, checkpoint

def subset_recall(pred_vec, gt_vec):
    """
    pred_vec, gt_vec: arrays of shape [num_classes] containing 0/1
    returns 1 or 0
    """
    gt_indices = np.where(gt_vec == 1)[0]
    for idx in gt_indices:
        if pred_vec[idx] != 1:
            return 0.0
    return 1.0

def subset_recall_macro(y_true, y_pred):
    scores = []
    for gt, pred in zip(y_true, y_pred):
        scores.append(subset_recall(pred, gt))
    return np.mean(scores)

import numpy as np

def label_recall_vector(pred_vec, gt_vec):
    """
    pred_vec, gt_vec: numpy arrays of shape [num_classes], values 0/1
    Returns recall in [0,1]
    """
    gt_indices = np.where(gt_vec == 1)[0]

    if len(gt_indices) == 0:
        return 1.0

    correct = 0
    for idx in gt_indices:
        if pred_vec[idx] == 1:
            correct += 1

    return correct / len(gt_indices)

def label_recall_macro(y_true, y_pred):
    """
    y_true: [N, C]  groundtruth multi-hot
    y_pred: [N, C]  predicted multi-hot
    """
    recalls = []
    for gt, pred in zip(y_true, y_pred):
        recalls.append(label_recall_vector(pred, gt))
    return np.mean(recalls)
