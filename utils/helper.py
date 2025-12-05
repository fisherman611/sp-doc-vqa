import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import torch
import os
import numpy as np
import re
from PIL import Image

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

def label_recall_vector(pred_vec, gt_vec):
    gt_idx = np.where(gt_vec == 1)[0]
    if len(gt_idx) == 0:
        return 1.0
    correct = sum(pred_vec[i] == 1 for i in gt_idx)
    return correct / len(gt_idx)

def label_recall_macro(Y_true, Y_pred):
    return np.mean([label_recall_vector(p, g) for p, g in zip(Y_pred, Y_true)])

def load_image_as_part(path: Path):
    """Load image file for Gemini API."""
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        img_bytes = f.read()
    return {
        "inline_data": {
            "mime_type": mime,
            "data": img_bytes
        }
    }


def clean_json_output(text: str):
    """Remove markdown fences and parse JSON object."""
    if not text:
        return {}
    
    # Remove markdown code blocks
    cleaned = re.sub(r"```json|```", "", text).strip()
    cleaned = re.sub(r"^json\n", "", cleaned, flags=re.IGNORECASE)
    
    try:
        return json.loads(cleaned)
    except Exception as e:
        print(f"JSON parse failed: {e}")
        print(f"Raw text: {cleaned[:200]}...")
        return {}


def load_ocr_info(ocr_filename: str, ocr_folder: Path) -> dict:
    """Load and extract text, words, and bounding boxes from OCR JSON file."""
    if not ocr_filename:
        return {"text": "", "lines": []}
    
    ocr_path = ocr_folder / ocr_filename
    if not ocr_path.exists():
        return {"text": "", "lines": []}
    
    try:
        with open(ocr_path, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
        
        # Extract text and bounding boxes from OCR data
        text_lines = []
        structured_lines = []
        
        if "recognitionResults" in ocr_data:
            for result in ocr_data["recognitionResults"]:
                if "lines" in result:
                    for line in result["lines"]:
                        if "text" in line:
                            line_info = {
                                "text": line["text"],
                                "boundingBox": line.get("boundingBox", [])
                            }
                            
                            # Extract words with their bounding boxes if available
                            words = []
                            if "words" in line:
                                for word in line["words"]:
                                    words.append({
                                        "text": word.get("text", ""),
                                        "boundingBox": word.get("boundingBox", [])
                                    })
                            line_info["words"] = words
                            
                            structured_lines.append(line_info)
                            text_lines.append(line["text"])
        
        return {
            "text": "\n".join(text_lines),
            "lines": structured_lines
        }
    except Exception as e:
        print(f"Error loading OCR file {ocr_filename}: {e}")
        return {"text": "", "lines": []}


def format_ocr_content(ocr_info: dict) -> str:
    """Format OCR data with text and bounding boxes for the prompt."""
    if not ocr_info["text"]:
        return ""
    
    ocr_formatted = "OCR text from document (with bounding boxes):\n\n"
    for idx, line_info in enumerate(ocr_info["lines"], 1):
        ocr_formatted += f"Line {idx}: \"{line_info['text']}\"\n"
        if line_info.get("boundingBox"):
            bbox = line_info["boundingBox"]
            ocr_formatted += f"  Bounding box: [{','.join(map(str, bbox))}]\n"
        
        # Include word-level bounding boxes if available
        if line_info.get("words"):
            ocr_formatted += "  Words:\n"
            for word_info in line_info["words"]:
                ocr_formatted += f"    - \"{word_info['text']}\""
                if word_info.get("boundingBox"):
                    bbox = word_info["boundingBox"]
                    ocr_formatted += f" [{','.join(map(str, bbox))}]"
                ocr_formatted += "\n"
        ocr_formatted += "\n"
    
    return ocr_formatted

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")