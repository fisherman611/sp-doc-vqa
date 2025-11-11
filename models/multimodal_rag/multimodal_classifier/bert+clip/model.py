import torch
import torch.nn as nn
from transformers import (
    AutoModel, 
    CLIPModel,
)
from utils.helper import load_config

config = load_config('config.json')

DEVICE = config['device']['cuda'] if torch.cuda.is_available() else config['device']['cpu']
TEXT_MODEL = config['text_model']
IMAGE_MODEL = config['image_model']
NUM_CLASSES = len(config['classes'])
MAX_LEN = config['max_len']

class BertCLIPMultimodalClassifier(nn.Module):
    def __init__(self, text_model_name=TEXT_MODEL, image_model_name=IMAGE_MODEL, num_classes=NUM_CLASSES) -> None:
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.image_encoder = CLIPModel.from_pretrained(image_model_name).vision_model
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, 512)
        self.image_proj = nn.Linear(self.image_encoder.config.hidden_size, 512)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, input_ids, attention_mask, image):
        txt_feat = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        img_feat = self.image_encoder(image).pooler_output
        txt_feat = self.text_proj(txt_feat)
        img_feat = self.image_proj(img_feat)
        fused = torch.cat([txt_feat, img_feat], dim=1)
        return self.classifier(fused)