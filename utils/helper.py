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