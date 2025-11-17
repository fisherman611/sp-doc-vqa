import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import *
from train_eval import *
from dataset import *
from utils.helper import load_config
from transformers import (
    AutoTokenizer,
    CLIPProcessor,
)
from torch.utils.data import DataLoader

config = load_config('models/multimodal_rag/multimodal_classifier/clip/config.json')
EPOCHS = config["epoch"]
BATCH_SIZE = config["batch_size"]
LEARNING_RATE = config["lr"]
MODEL = config["image_model"]

clip_proc = CLIPProcessor.from_pretrained(MODEL, use_fast=True)

label_list = config['classes']
label2id = {lbl: i for i, lbl in enumerate(label_list)}

train_ds = CLIPDocVQAMultimodalDataset("data/spdocvqa_qas/train_v1.0_withQT.json", "data/spdocvqa_images", clip_proc.image_processor, label2id)
val_ds = CLIPDocVQAMultimodalDataset("data/spdocvqa_qas/val_v1.0_withQT.json", "data/spdocvqa_images", clip_proc.image_processor, label2id)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

model = CLIPMultimodalClassifier(MODEL, NUM_CLASSES)
train_model(model, train_loader, val_loader, label2id, epochs=EPOCHS, lr=LEARNING_RATE)