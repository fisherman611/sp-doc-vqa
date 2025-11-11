from model import *
from train_eval import *
from dataset import *
from utils.helper import load_config
from transformers import (
    AutoTokenizer,
    CLIPProcessor,
)
from torch.utils.data import DataLoader

config = load_config("config.json")
EPOCHS = config["epoch"]
BATCH_SIZE = config["batch_size"]
LEARNING_RATE = config["lr"]
TEXT_MODEL = config["text_model"]
IMAGE_MODEL = config["image_model"]

tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
clip_proc = CLIPProcessor.from_pretrained(IMAGE_MODEL)

label_list = config['classes']
label2id = {lbl: i for i, lbl in enumerate(label_list)}

train_ds = BertCLIPDocVQAMultimodalDataset("train.csv", "documents", tokenizer, clip_proc.image_processor, label2id)
val_ds = BertCLIPDocVQAMultimodalDataset("val.csv", "documents", tokenizer, clip_proc.image_processor, label2id)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

model = MultimodalClassifier(TEXT_MODEL, IMAGE_MODEL, NUM_CLASSES)
train_model(model, train_loader, val_loader, label2id, epochs=EPOCHS, lr=LEARNING_RATE)