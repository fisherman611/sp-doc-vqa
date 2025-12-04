import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(PROJECT_ROOT / "utils")

import time
import re
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from tqdm.auto import tqdm
from utils.helper import *

# ==========================
# CONFIG
# ==========================
load_dotenv()
with open("models/multimodal_retrieval_cot/description_generator/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
    
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = config["model_name"]

IMAGE_FOLDER = Path("data/spdocvqa_images")

# Rate limits
MAX_RPM = config["max_rpm"]
MAX_RPD = config["max_rpd"]
MAX_TPM = config["max_tpm"]
AVG_TOKENS_PER_CALL = config["avg_tokens_per_call"]
SLEEP_BETWEEN_CALLS = 60.0 / MAX_RPM

# Load system prompt
with open("models/multimodal_retrieval_cot/description_generator/system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()
    
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(
    MODEL_NAME,
    system_instruction=SYSTEM_PROMPT.strip()
)

def generate_description(
    image_path: Path,
    question: str,
    ocr_filename: str = "",
    ocr_folder: Path = None,
    timeout: int = 180
) -> dict:
    """
    Generate image description for a single sample.
    
    Args:
        image_path: Path to the document image
        question: The question being asked
        ocr_filename: Optional OCR JSON filename
        ocr_folder: Optional path to OCR folder (defaults to data/spdocvqa_ocr)
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary with:
        - image_description: Generated description of the image
        - success: Boolean indicating if generation was successful
        - error: Error message if success is False
    """
    if ocr_folder is None:
        ocr_folder = Path("data/spdocvqa_ocr")
    
    # Check if image exists
    if not image_path.exists():
        return {
            "image_description": "",
            "success": False,
            "error": f"Image not found: {image_path}"
        }
    
    # Load OCR info if available
    ocr_info = load_ocr_info(ocr_filename, ocr_folder)
    
    # Load image
    try:
        image_part = load_image_as_part(image_path)
    except Exception as e:
        return {
            "image_description": "",
            "success": False,
            "error": f"Error loading image: {e}"
        }
    
    # Build the content list
    content_parts = [
        {"text": f"Question: {question}"}
    ]
    
    # Add OCR text with bounding boxes if available
    if ocr_info["text"]:
        ocr_formatted = format_ocr_content(ocr_info)
        content_parts.append({"text": ocr_formatted})
    
    content_parts.extend([
        {"text": "Document image:"},
        image_part
    ])
    
    # Generate content
    try:
        response = model.generate_content(
            content_parts,
            request_options={"timeout": timeout}
        )
        
        raw_text = response.text.strip() if response.text else ""
        parsed = clean_json_output(raw_text)
        
        image_description = parsed.get("image_description", "")
        
        return {
            "image_description": image_description,
            "success": True,
            "error": None
        }
        
    except Exception as e:
        return {
            "image_description": "",
            "success": False,
            "error": f"Error generating content: {e}"
        }


class DescriptionGenerator:
    """
    A class wrapper for generating image descriptions.
    Useful for integration into inference pipelines.
    """
    
    def __init__(self, ocr_folder: Path = None, default_timeout: int = 180):
        """
        Initialize the description generator.
        
        Args:
            ocr_folder: Path to OCR folder (defaults to data/spdocvqa_ocr)
            default_timeout: Default timeout for API requests
        """
        self.ocr_folder = ocr_folder or Path("data/spdocvqa_ocr")
        self.default_timeout = default_timeout
        self.model = model
    
    def generate(
        self,
        image_path: Path,
        question: str,
        ocr_filename: str = "",
        timeout: int = None
    ) -> dict:
        """
        Generate description for a sample.
        
        Args:
            image_path: Path to the document image
            question: The question being asked
            ocr_filename: Optional OCR JSON filename
            timeout: Request timeout (uses default if None)
            
        Returns:
            Dictionary with generation results
        """
        if timeout is None:
            timeout = self.default_timeout
            
        return generate_description(
            image_path=image_path,
            question=question,
            ocr_filename=ocr_filename,
            ocr_folder=self.ocr_folder,
            timeout=timeout
        )
    
    def generate_from_sample(self, sample: dict, image_folder: Path = None) -> dict:
        """
        Generate description from a sample dictionary.
        
        Args:
            sample: Dictionary containing 'image', 'question', and optionally 'ocr'
            image_folder: Path to image folder (defaults to data/spdocvqa_images)
            
        Returns:
            Dictionary with generation results
        """
        if image_folder is None:
            image_folder = IMAGE_FOLDER
        
        image_path = image_folder / Path(sample["image"]).name
        question = sample.get("question", "")
        ocr_filename = sample.get("ocr", "")
        
        return self.generate(
            image_path=image_path,
            question=question,
            ocr_filename=ocr_filename
        )

if __name__ == "__main__":
    sample = {
            "questionId": 49153,
            "question": "What is the ‘actual’ value per 1000, during the year 1975?",
            "question_types": [
                "figure/diagram"
            ],
            "image": "documents/pybv0228_81.png",
            "docId": 14465,
            "ucsf_document_id": "pybv0228",
            "ucsf_document_page_no": "81",
            "answers": [
                "0.28"
            ],
            "data_split": "val",
            "ocr": "pybv0228_81.json"
        }
    description_generator = DescriptionGenerator()
    description = description_generator.generate_from_sample(sample=sample, image_folder=IMAGE_FOLDER)
    print(description)