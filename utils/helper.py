import json
import torch
import os

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
    print(f"Metrics: MacroF1={checkpoint['best_macro_f1']:.4f}, "
          f"MicroF1={checkpoint['micro_f1']:.4f}, "
          f"SubsetAcc={checkpoint['subset_acc']:.4f}")
    
    return model, checkpoint

def label_recall(pred, gt):
    """
    pred: list of predicted labels (strings)
    gt:   list of groundtruth labels (strings)

    Returns recall in [0,1].
    """
    pred_set = set(pred)
    gt_set = set(gt)

    if len(gt_set) == 0:
        return 1.0

    intersection = pred_set & gt_set
    return len(intersection) / len(gt_set)

def label_recall_micro(preds, gts):
    """
    preds: list of predicted label lists
    gts:   list of groundtruth label lists

    Micro recall = total correct / total groundtruth labels
    """
    total_correct = 0
    total_gt = 0

    for pred, gt in zip(preds, gts):
        pred_set = set(pred)
        gt_set = set(gt)
        total_correct += len(pred_set & gt_set)
        total_gt += len(gt_set)

    if total_gt == 0:
        return 1.0

    return total_correct / total_gt


def label_recall_macro(preds, gts):
    scores = []
    for pred, gt in zip(preds, gts):
        scores.append(label_recall(pred, gt))
    return sum(scores) / len(scores)
