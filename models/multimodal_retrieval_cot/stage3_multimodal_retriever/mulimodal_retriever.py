import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(PROJECT_ROOT / "utils")
sys.path.append(PROJECT_ROOT / "models/multimodal_retrieval_cot/stage3_multimodal_retriever")

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
from multimodal_biencoder import MultimodalBiEncoder
from cross_encoder_reranker import CrossEncoderReranker

load_dotenv()
from huggingface_hub import login
login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))

with open("models/multimodal_retrieval_cot/stage3_multimodal_retriever/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

TOP_K = config["top_k"]
BATCH_SIZE = config["batch_size"]

class MultimodalRetriever:
    """
    Full pipeline:
    - build embeddings with multimodal bi-encoder
    - index with FAISS
    - retrieve top-k
    - rerank with cross-encoder
    """
    def __init__(
        self, 
        multimodal_biencoder: MultimodalBiEncoder,
        reranker: CrossEncoderReranker,
        faiss_metric: str="ip"
    ) -> None:
        self.multimodal_biencoder = multimodal_biencoder
        self.reranker = reranker
        self.index = None
        self.examples_meta = []
        
        if faiss_metric == "ip":
            self.metric = faiss.METRIC_INNER_PRODUCT
        elif faiss_metric == "l2":
            self.metric = faiss.METRIC_L2
        else:
            raise ValueError("faiss_metric must be 'ip' or 'l2'")
    
    def build_index(
        self,
        examples: List[Dict],
        batch_size: int = BATCH_SIZE,
        use_gpu: bool = False,
    ) -> None:
        """
        Build FAISS index from examples using the multimodal bi-encoder.
        
        Args:
            examples: List of example dicts with keys: question, image, image_description, etc.
            batch_size: Batch size for encoding
            use_gpu: Whether to use GPU for FAISS indexing
        """
        print(f"Building index from {len(examples)} examples...")
        
        all_embeddings = []
        self.examples_meta = []
        
        # Process in batches
        for start_idx in range(0, len(examples), batch_size):
            end_idx = min(start_idx + batch_size, len(examples))
            batch = examples[start_idx:end_idx]
            
            # Prepare batch inputs
            images = [self.multimodal_biencoder.build_image_input(ex) for ex in batch]
            texts = [self.multimodal_biencoder.build_text_input(ex) for ex in batch]
            
            # Encode using bi-encoder
            embeddings = self.multimodal_biencoder.encode_pair(images, texts)
            all_embeddings.append(embeddings.cpu().numpy())
            
            # Store metadata
            self.examples_meta.extend(batch)
            
            if (start_idx // batch_size + 1) % 10 == 0:
                print(f"Processed {end_idx}/{len(examples)} examples")
        
        # Concatenate all embeddings
        all_embeddings = np.vstack(all_embeddings).astype('float32')
        embedding_dim = all_embeddings.shape[1]
        
        print(f"Embedding shape: {all_embeddings.shape}")
        
        # Build FAISS index
        if self.metric == faiss.METRIC_INNER_PRODUCT:
            # Normalize for inner product
            faiss.normalize_L2(all_embeddings)
            self.index = faiss.IndexFlatIP(embedding_dim)
        else:
            self.index = faiss.IndexFlatL2(embedding_dim)
        
        # Use GPU if requested
        if use_gpu and torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        # Add vectors to index
        self.index.add(all_embeddings)
        
        print(f"Index built with {self.index.ntotal} vectors")
    
    def retrieve(
        self,
        query_example: Dict,
        top_k: int = TOP_K,
    ) -> List[Dict]:
        """
        Retrieve top-k most similar examples using FAISS.
        
        Args:
            query_example: Query dict with keys: question, image, image_description
            top_k: Number of candidates to retrieve
            
        Returns:
            List of top-k candidate examples
        """
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("Index is not built. Call build_index() first.")
        
        # Encode query
        query_image = self.multimodal_biencoder.build_image_input(query_example)
        query_text = self.multimodal_biencoder.build_text_input(query_example)
        
        query_emb = self.multimodal_biencoder.encode_pair([query_image], [query_text])
        query_emb = query_emb.cpu().numpy().astype('float32')
        
        # Normalize for inner product
        if self.metric == faiss.METRIC_INNER_PRODUCT:
            faiss.normalize_L2(query_emb)
        
        # Search in FAISS index
        distances, indices = self.index.search(query_emb, min(top_k, self.index.ntotal))
        
        # Retrieve candidates
        candidates = []
        for idx in indices[0]:
            if idx >= 0 and idx < len(self.examples_meta):
                candidates.append(self.examples_meta[idx])
        
        return candidates
    
    def retrieve_and_rerank(
        self,
        query_example: Dict,
        top_k: int = 10,
        rerank_top_k: int = 5,
        batch_size: int = 8,
    ) -> List[Tuple[Dict, float]]:
        """
        Full retrieval pipeline: retrieve with bi-encoder, then rerank with cross-encoder.
        
        Args:
            query_example: Query dict
            top_k: Number of candidates to retrieve from FAISS
            rerank_top_k: Number of top candidates to return after reranking
            batch_size: Batch size for reranking
            
        Returns:
            List of (example, score) tuples sorted by relevance
        """
        # Step 1: Retrieve candidates using bi-encoder + FAISS
        candidates = self.retrieve(query_example, top_k=top_k)
        
        if not candidates:
            return []
        
        # Step 2: Rerank using cross-encoder
        reranked = self.reranker.rerank(
            query_example=query_example,
            candidate_examples=candidates,
            batch_size=batch_size,
        )
        
        # Return top rerank_top_k
        return reranked[:rerank_top_k]
    
    def save_index(self, index_path: str, metadata_path: str) -> None:
        """
        Save FAISS index and metadata to disk.
        
        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save examples metadata (JSON)
        """
        if self.index is None:
            raise ValueError("No index to save. Build index first.")
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.examples_meta, f, ensure_ascii=False, indent=2)
        
        print(f"Index saved to {index_path}")
        print(f"Metadata saved to {metadata_path}")
    
    def load_index(self, index_path: str, metadata_path: str, use_gpu: bool = False) -> None:
        """
        Load FAISS index and metadata from disk.
        
        Args:
            index_path: Path to FAISS index
            metadata_path: Path to examples metadata (JSON)
            use_gpu: Whether to move index to GPU
        """
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Move to GPU if requested
        if use_gpu and torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.examples_meta = json.load(f)
        
        print(f"Index loaded from {index_path} with {self.index.ntotal} vectors")
        print(f"Metadata loaded from {metadata_path} with {len(self.examples_meta)} examples")


if __name__ == "__main__":
    # Example usage
    print("Initializing models...")
    
    # Initialize bi-encoder
    biencoder = MultimodalBiEncoder(
        text_model_name="sentence-transformers/all-mpnet-base-v2",
        image_model_name="openai/clip-vit-base-patch32",
        proj_dim=768,
    )
    
    # Initialize cross-encoder reranker
    reranker = CrossEncoderReranker(
        model_name="Salesforce/blip2-itm-vit-g-coco",
        fp16=True,
    )
    
    # Initialize retriever
    retriever = MultimodalRetriever(
        multimodal_biencoder=biencoder,
        reranker=reranker,
        faiss_metric="ip",
    )
    
    print("Multimodal Retriever initialized successfully!")
    