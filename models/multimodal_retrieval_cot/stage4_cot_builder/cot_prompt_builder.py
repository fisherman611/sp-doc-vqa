import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(PROJECT_ROOT / "utils")

from PIL import Image
import json
from typing import List, Dict, Any, Optional
from utils.helper import (
    load_image,
    build_example_template, 
    build_query_template
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
        self.full_prompt = ""
                
    def build(self):
        prompt_blocks = []
        prompt_images = []
        
        for idx, ex in enumerate(self.retrieved_examples[:self.max_examples], start=1):
            # Load image
            img = Image.open(ex["image"]).convert("RGB")
            prompt_images.append(img)

            # Build text block
            block_text = build_example_template(ex, idx)
            prompt_blocks.append(block_text)
        
        query_image_slot = len(prompt_images) + 1

        q_img = Image.open(self.query_ex["image"]).convert("RGB")
        prompt_images.append(q_img)

        query_text = build_query_template(self.query_ex, self.ocr_root, idx=query_image_slot)

        # ----- Build final prompt text -----
        prompt_text = "\n".join(prompt_blocks) + "\nNow answer the query example:\n" + query_text

        return prompt_text, prompt_images
        
    def length(self):
        return len(self.full_prompt)
    
# if __name__ == "__main__":
#     ex = {
#         "questionId": 337,
#         "question": "what is the date mentioned in this letter?",
#         "question_types": [
#             "handwritten",
#             "form"
#         ],
#         "image": "data\\spdocvqa_images\\xnbl0037_1.png",
#         "docId": 279,
#         "ucsf_document_id": "xnbl0037",
#         "ucsf_document_page_no": "1",
#         "ocr": "xnbl0037_1.json",
#         "answers": [
#             "1/8/93"
#         ],
#         "image_description": "The document is a form or letter titled \"Confidential\" and \"RJRT PR APPROVAL\" at the top. It contains several structured fields in the top-left section, including 'DATE:', 'SUBJECT:', 'PROPOSED RELEASE DATE:', 'FOR RELEASE TO:', and 'CONTACT:'. A handwritten date, '1/8/93', is visible next to the 'DATE:' label. Further down, there is a list of names under a 'ROUTE TO:' section, and a table-like structure on the right with headers 'Initials' and 'Date', under which '1/8/93' is also visibly written.",
#         "answer_explanation": "Step 1: Located the label 'DATE :' with bounding box [254,295,343,294,344,322,255,323] in the top-left section of the document. Step 2: Identified the date value next to this label. The OCR text for this value is '1/8/13' with bounding box [396,262,561,262,559,333,398,327]. Visually, the date written is '1/8/93'.",
#         "reasoning_type": "single-hop"
#     }
#     cot_prompt_builder = CoTPromptBuilder(query_ex=ex, retrieved_examples=[ex], ocr_root="data/spdocvqa_ocr")
#     print(*cot_prompt_builder.build(), end="\n\n")