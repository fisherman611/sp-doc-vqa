import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    AutoTokenizer,
    CLIPProcessor,
)
import numpy as np
from tqdm.auto import tqdm
import json
from pathlib import Path
from utils.helper import load_config

config = load_config('config.json')

DEVICE = config['device']['cuda'] if torch.cuda.is_available() else config['device']['cpu']
TEXT_MODEL = config['text_model']
IMAGE_MODEL = config['image_model']
NUM_CLASSES = len(config['classes'])
MAX_LEN = config['max_len']

class BertCLIPDocVQAMultimodalDataset(Dataset):
    def __init__(self, json_path, img_root, tokenizer, preprocess, label2id, max_len=MAX_LEN) -> None:
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.questions = [self.data['data'][i]['question'].strip() for i in range(len(self.data['data']))]
        # self.questions = [self.data['data'][i]['question'].strip() for i in range(1)]
        self.image_paths = [Path(self.data['data'][i]['image']).name for i in range(len(self.data['data']))]
        # self.image_paths = [Path(self.data['data'][i]['image']).name for i in range(1)]
        self.labels = []
        for x in tqdm(self.data['data']):
        # for x in tqdm(self.data['data'][:1]):
            vec = np.zeros(len(label2id), dtype=np.float32)
            for lbl in x['question_types']:
                vec[label2id[lbl]] = 1.0
            self.labels.append(vec)
            
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.img_root = Path(img_root)
        self.max_len = max_len
        
        
    def __len__(self):
        return len(self.questions)
    
    
    def __getitem__(self, idx):
        q = self.questions[idx]
        enc = self.tokenizer(
            q,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        img_path = self.img_root / self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.preprocess(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "image": image_tensor,
            "label": torch.tensor(self.labels[idx], dtype=torch.float)
        }
        
# if __name__ == "__main__":
#     tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, use_fast=True)
#     clip_proc = CLIPProcessor.from_pretrained(IMAGE_MODEL, use_fast=True)

#     label_list = ["handwritten", "form", "layout", "table/list", "others", "free_text", "Image/Photo", "figure/diagram", "Yes/No"]
#     label2id = {lbl: i for i, lbl in enumerate(label_list)}

#     train_ds = BertCLIPDocVQAMultimodalDataset("data/spdocvqa_qas/val_v1.0_withQT.json", "data/spdocvqa_images", tokenizer, clip_proc.feature_extractor, label2id)
#     print(len(train_ds))
#     print(train_ds[0])    