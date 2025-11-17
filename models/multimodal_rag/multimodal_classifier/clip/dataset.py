import torch
from torch.utils.data import Dataset
import torch.nn as nn
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from tqdm.auto import tqdm
import json
from pathlib import Path
# from utils.helper import load_config

# config = load_config('config.json')

# # DEVICE = config['device']['cuda'] if torch.cuda.is_available() else config['device']['cpu']
# # MODEL = config['model']
# # NUM_CLASSES = len(config['classes'])
# # MAX_LEN = config['max_len']
MAX_LEN = 64

class CLIPDocVQAMultimodalDataset(Dataset):
    def __init__(self, json_path, img_root, processor, label2id, max_len=MAX_LEN) -> None:
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        # self.questions = [self.data['data'][i]['question'].strip() for i in range(len(self.data['data']))]
        self.questions = [self.data['data'][i]['question'].strip() for i in range(1)]
        # self.image_paths = [Path(self.data['data'][i]['image']).name for i in range(len(self.data['data']))]
        self.image_paths = [Path(self.data['data'][i]['image']).name for i in range(1)]
        self.labels = []
        # for x in tqdm(self.data['data']):
        for x in tqdm(self.data['data'][:1]):
            vec = np.zeros(len(label2id), dtype=np.float32)
            for lbl in x['question_types']:
                vec[label2id[lbl]] = 1.0
            self.labels.append(vec)
            
        self.processor = processor
        self.img_root = Path(img_root)
        self.max_len = max_len
        
    
    def __len__(self):
        return len(self.questions)
    
    
    def __getitem__(self, idx):
        q = self.questions[idx]
        img_path = self.img_root / self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        enc = self.processor(
            text=q,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )
        
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item
    
# if __name__ == "__main__":
#     processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

#     label_list = ["handwritten", "form", "layout", "table/list", "others", "free_text", "Image/Photo", "figure/diagram", "Yes/No"]
#     label2id = {lbl: i for i, lbl in enumerate(label_list)}

#     train_ds = CLIPDocVQAMultimodalDataset("data/spdocvqa_qas/val_v1.0_withQT.json", "data/spdocvqa_images", processor=processor, label2id=label2id)
#     print(len(train_ds))
#     print(train_ds[0])  