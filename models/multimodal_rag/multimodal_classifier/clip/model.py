import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import f1_score, average_precision_score
from tqdm import tqdm
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_LABELS = 9  # adjust to your total label count
LABELS = [
    "handwritten", "form", "layout", "table/list", "others",
    "free_text", "Image/Photo", "figure/diagram", "Yes/No"
]
JSON_PATH = "data/spdocvqa_qas/val_v1.0_withQT.json"
IMAGE_FOLDER = "documents/"
BATCH_SIZE = 8
EPOCHS = 5
LR = 2e-5
MODEL = "openai/clip-vit-base-patch32"

class CLIPMultimodalClassifier(nn.Module):
    def __init__(self, model_name=MODEL, num_labels=NUM_LABELS) -> None:
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        emb_dim = self.clip.text_projection.shape[1]
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels)
        )


    def forward(self, input_ids, attention_mask, pixel_values):
        outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
        # combine text + image embeddings (normalized)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        fused = torch.cat([image_embeds, text_embeds], dim=1)
        logits = self.classifier(fused)
        return logits
