import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(PROJECT_ROOT / "utils")

import json
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image
import faiss
import torch
import torch.nn as nn
from transformers import Blip2Processor, Blip2ForImageTextRetrieval
from utils.helper import load_image
from dotenv import load_dotenv

load_dotenv()
from huggingface_hub import login
login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))

class CrossEncoderReranker(nn.Module):
    """
    Cross-encoder reranker using BLIP-2 + Q-Former

    Given:
        - query_example: { "question", "image_description", "image" }
        - candidate_examples: list of same-structure dicts
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip2-itm-vit-g-coco",
        device: str = None,
        fp16: bool = False,
    ) -> None:
        super().__init__()

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.processor = Blip2Processor.from_pretrained(model_name)
        if fp16 and self.device == "cuda":
            self.model = Blip2ForImageTextRetrieval.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
            )
        else:
            self.model = Blip2ForImageTextRetrieval.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

    # ---------- text/image building helpers ----------

    @staticmethod
    def build_text_input(example: Dict) -> str:
        """
        Combine question and image_description into a single text.
        """
        question = example.get("question", "")
        img_description = example.get("image_description", "")
        return f"Question: {question} [SEP] Image description: {img_description}"

    @classmethod
    def build_pair_text(cls, query_example: Dict, cand_example: Dict) -> str:
        """
        Joint text for (query triple, candidate triple).

        You can tweak this prompt to change behavior.
        """
        q_txt = cls.build_text_input(query_example)
        c_txt = cls.build_text_input(cand_example)

        # Cross-encoder style: both query and candidate in the same text.
        return (
            f"[QUERY] {q_txt} "
            f"[SEP] [CANDIDATE] {c_txt} "
            f"[TASK] Does this candidate match the query?"
        )

    @staticmethod
    def build_image_input(example: Dict) -> Image.Image:
        image_path = example.get("image", "")
        return load_image(image_path)

    # ---------- scoring & reranking ----------

    @torch.no_grad()
    def score_candidates(
        self,
        query_example: Dict,
        candidate_examples: List[Dict],
        batch_size: int = 8,
    ) -> List[float]:
        """
        Compute BLIP-2 ITM scores for all candidates, conditioned on query_example.

        Returns:
            List[float]: probability-like scores (higher = more relevant).
        """
        self.model.eval()

        all_scores: List[float] = []
        num_cands = len(candidate_examples)

        for start in range(0, num_cands, batch_size):
            end = min(start + batch_size, num_cands)
            batch = candidate_examples[start:end]

            images = [self.build_image_input(ex) for ex in batch]
            texts = [self.build_pair_text(query_example, ex) for ex in batch]

            inputs = self.processor(
                images=images,
                text=texts,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            outputs = self.model(**inputs)

            if hasattr(outputs, "itm_score"):
                logits = outputs.itm_score  # [B, 1] or [B]
            else:
                logits = outputs[0]  # fallback: first element

            # Ensure shape [B]
            logits = logits.view(-1)

            # Convert logits -> probabilities with sigmoid
            probs = torch.sigmoid(logits)

            all_scores.extend(probs.detach().cpu().tolist())

        return all_scores

    @torch.no_grad()
    def rerank(
        self,
        query_example: Dict,
        candidate_examples: List[Dict],
        batch_size: int = 8,
    ) -> List[Tuple[Dict, float]]:
        """
        Rerank candidate_examples by relevance to query_example.

        Returns:
            List of (candidate_example, score) sorted by score desc.
        """
        scores = self.score_candidates(
            query_example=query_example,
            candidate_examples=candidate_examples,
            batch_size=batch_size,
        )

        ranked = sorted(
            zip(candidate_examples, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked