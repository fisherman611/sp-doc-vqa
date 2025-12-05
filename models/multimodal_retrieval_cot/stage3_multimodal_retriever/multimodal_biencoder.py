import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(PROJECT_ROOT / "utils")

import json
from typing import List, Dict, Any, Tuple
import numpy as np
from PIL import Image
import faiss
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from utils.helper import load_image
from dotenv import load_dotenv

load_dotenv()
from huggingface_hub import login
login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))

class MultimodalBiEncoder:
    """
    Multimodal Bi-encoder for (image, question, image_description)
    """
    def __init__(
        self,
        text_model_name: str ="sentence-transformers/all-mpnet-base-v2",
        image_model_name: str ="openai/clip-vit-base-patch32",
        proj_dim: int=768,
        device: str=None
    ) -> None:
        super().__init__()
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Text encoder
        self.text_model = SentenceTransformer(text_model_name)
        self.text_model.to(self.device)
        text_dim = self.text_model.get_sentence_embedding_dimension()
        
        # Image encoder
        self.image_model = CLIPModel.from_pretrained(image_model_name)
        self.image_model.to(self.device)
        self.image_processor = CLIPProcessor.from_pretrained(image_model_name)
        image_dim = self.image_model.visual_projection.out_features
        
        self.proj = nn.Linear(image_dim + text_dim, proj_dim).to(self.device)
        
    @staticmethod
    def build_text_input(example: Dict) -> str:
        """
        Combine question and image_description into a single text.
        """
        question = example.get("question", "")
        img_description = example.get("image_description", "")
        return f"Question: {question} [SEP] Image description: {img_description}"
    
    @staticmethod
    def build_image_input(example: Dict) -> Image.Image:
        image_path = example.get("image", "")
        return load_image(image_path)
    
    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts using SentenceTransformers
        return normalized embeddings: [B, text_dim]
        """
        emb = self.text_model.encode(
            texts,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=False,
        )
        
        return emb
    
    @torch.no_grad()
    def encode_image(self, images: List[Image.Image]) -> torch.Tensor:
        """  
        Encode images using CLIP visual encoder
        return normalized embeddings [B, image_dim]
        """
        inputs = self.image_processor(images=images, return_tensor="pt").to(self.device)
        vision_out = self.image_model.get_image_features(**inputs)
        vision_out = vision_out / vision_out.norm(dim=-1, keepdim=True)
        return vision_out
    
    @torch.no_grad()
    def encode_pair(
        self,
        images: List[Image.Image],
        texts: List[str]
    ) -> torch.Tensor:
        """  
        Encode (image, text) pairs into a shared embedding space
        Returns normalized embeddings: [B, proj_dim]
        """
        img_emb = self.encode_image(images)
        txt_emb = self.encode_text(texts)
        concat = torch.cat([img_emb, txt_emb], dim=-1)
        proj = self.proj(concat)
        proj = proj / proj.norm(dim=-1, keepdim=True)
        return proj
    
    @staticmethod
    def info_nce_loss(
        query_emb: torch.Tensor,
        doc_emb: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """
        Standard InfoNCE loss for symmetric contrastive learning.
        query_emb: [B, D]
        doc_emb:   [B, D]
        """
        logits = query_emb @ doc_emb.t() / temperature  # [B, B]
        labels = torch.arange(query_emb.size(0), device=query_emb.device)
        loss_q = torch.nn.functional.cross_entropy(logits, labels)
        loss_d = torch.nn.functional.cross_entropy(logits.t(), labels)
        return (loss_q + loss_d) / 2.0

    def contrastive_training_step(
        self,
        batch_examples: List[Dict],
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """
        Simple contrastive training step.
        You can customize what is 'query' and 'doc'.
        """
        self.train()

        images_q, texts_q = [], []
        images_d, texts_d = [], []

        for ex in batch_examples:
            img = load_image(ex["image"])

            # Query: image + (question + image_description)
            q_text = self.build_text_input(ex)

            # Doc: include answer + explanation
            a = ex.get("answers", [""])[0]
            exp = ex.get("answer_explanation", "")
            d_text = (
                f"Question: {ex.get('question', '')} "
                f"Answer: {a} "
                f"Explanation: {exp}"
            )

            images_q.append(img)
            texts_q.append(q_text)
            images_d.append(img)
            texts_d.append(d_text)

        query_emb = self.encode_pair(images_q, texts_q)  # [B, D]
        doc_emb = self.encode_pair(images_d, texts_d)    # [B, D]

        loss = self.info_nce_loss(query_emb, doc_emb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return float(loss.item())