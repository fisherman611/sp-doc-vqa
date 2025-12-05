import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(PROJECT_ROOT / "utils")

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from huggingface_hub import login
from qwen_vl_utils import process_vision_info
from PIL import Image
import json
from typing import List, Dict, Any, Optional
from utils.helper import extract_final_answer, normalize_answer, majority_vote
from dotenv import load_dotenv
from stage4_cot_builder.cot_prompt_builder import *

load_dotenv()

### Login to Hugging Face Hub
from huggingface_hub import login

login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))


class MLLMInference:
    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        device: str = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ) -> None:
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        if device is None:
            self.device = "cuda" if torch.cuda.is_available else "cpu"
        else:
            self.device = device

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=torch.float16, device_map="auto"
        ).eval()

        if hasattr(self.model, "generation_config"):
            self.model.generation_config.use_cache = False

        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def build_messages(self, prompt_text: str, images) -> List[Dict[str, Any]]:
        user_content = []
        for img in images:
            user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": prompt_text})
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        return messages
        
    
    def prepare_input(self, messages):
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        return inputs

    def single_generate(self, inputs) -> str:
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens
            )

        trimmed_ids = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        pred = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return pred