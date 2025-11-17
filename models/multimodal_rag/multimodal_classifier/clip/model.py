import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import f1_score, average_precision_score
from tqdm import tqdm
from PIL import Image
from utils.helper import load_config

config = load_config('models/multimodal_rag/multimodal_classifier/clip/config.json')

DEVICE = config['device']['cuda'] if torch.cuda.is_available() else config['device']['cpu']
MODEL = config['model']
NUM_CLASSES = len(config['classes'])
MAX_LEN = config['max_len']

class CLIPMultimodalClassifier(nn.Module):
    def __init__(self, model_name=MODEL, num_labels=NUM_CLASSES, freeze_clip=True) -> None:
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        emb_dim = self.clip.config.projection_dim

        fused_dim = emb_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_labels)
        )

        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False


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
