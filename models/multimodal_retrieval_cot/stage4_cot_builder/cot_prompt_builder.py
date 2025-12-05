import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(PROJECT_ROOT / "utils")

import json
from typing import List, Dict, Any, Optional
from utils.helper import (
    load_ocr_text_from_file, 
    build_example_block, 
    build_query_block
)

class CoTPromptBuilder:
    def __init__(
        self,
        query_ex: Dict[str, Any],
        retrieved_examples: List[Dict[str, Any]],
        ocr_root: str,
        max_examples: int=5,
    ) -> None:
        self.query_ex = query_ex
        self.retrieved_examples = retrieved_examples
        self.ocr_root = ocr_root
        self.max_examples = max_examples
        self.header = """You are a visual document question answering assistant.
You are given:
- A document IMAGE (path is provided so the system can load it)
- OCR_TEXT extracted from the document
- An IMAGE_DESCRIPTION summarizing the document
- A QUESTION about the document

Your task:
1. Carefully reason step by step using the OCR_TEXT and the document layout.
2. Explain your reasoning in [CHAIN_OF_THOUGHT].
3. Then provide the short final answer in [FINAL_ANSWER], usually a span from the document.

Here are some examples:

"""
        self.full_prompt = ""
                
    def build(self):
        example_blocks = []
        for i, ex in enumerate(self.retrieved_examples[:self.max_examples], start=1):
            example_blocks.append(build_example_block(ex, i))
        
        example_str = "\n\n".join(example_blocks)
        
        query_block = build_query_block(self.query_ex, self.ocr_root)
        full_prompt = self.header + example_str + "\n\nNow answer the query example.\n\n" + query_block
        self.full_prompt = full_prompt
        return full_prompt
    
    def length(self):
        return len(self.full_prompt)