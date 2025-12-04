import os
import json
import time
import re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from tqdm.auto import tqdm

# ==========================
# CONFIG
# ==========================
load_dotenv()
with open("augment_data/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = config["model_name"]

# Dataset selection - change to use train or val
DATASET_SPLIT = config["dataset_split"]  # "train" or "val"

IMAGE_FOLDER = Path("data/spdocvqa_images")
DATA_PATH = Path(f"data/spdocvqa_qas/{DATASET_SPLIT}_v1.0_withQT_ocr.json")
OCR_FOLDER = Path("data/spdocvqa_ocr")
OUTPUT_PATH = Path(f"data/augmented_data/gemini_augmented_{DATASET_SPLIT}_descriptions_explanations.json")

# Rate limits
MAX_RPM = config["max_rpm"]
MAX_RPD = config["max_rpd"]
MAX_TPM = config["max_tpm"]
AVG_TOKENS_PER_CALL = config["avg_tokens_per_call"]  # Higher since we're including OCR text + generating more content
SLEEP_BETWEEN_CALLS = 60.0 / MAX_RPM

# Load system prompt
with open('augment_data/gen_explaination_and_description_prompt.txt', 'r', encoding='utf-8') as f:
    SYSTEM_PROMPT = f.read()

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(
    MODEL_NAME,
    system_instruction=SYSTEM_PROMPT.strip()
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_image_as_part(path: Path):
    """Load image file for Gemini API."""
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        img_bytes = f.read()
    return {
        "inline_data": {
            "mime_type": mime,
            "data": img_bytes
        }
    }


def clean_json_output(text: str):
    """Remove markdown fences and parse JSON object."""
    if not text:
        return {}
    
    # Remove markdown code blocks
    cleaned = re.sub(r"```json|```", "", text).strip()
    cleaned = re.sub(r"^json\n", "", cleaned, flags=re.IGNORECASE)
    
    try:
        return json.loads(cleaned)
    except Exception as e:
        print(f"JSON parse failed: {e}")
        print(f"Raw text: {cleaned[:200]}...")
        return {}


def load_ocr_info(ocr_filename: str, ocr_folder: Path) -> dict:
    """
    Load and extract text, words, and bounding boxes from OCR JSON file.
    
    This OCR data is crucial for generating accurate image descriptions as it provides:
    - Readable text content from the document
    - Spatial information (bounding boxes) for locating text
    - Word-level details for precise referencing
    
    Returns:
        dict with 'text' (full text) and 'lines' (structured data with bounding boxes)
    """
    if not ocr_filename:
        return {"text": "", "lines": []}
    
    ocr_path = ocr_folder / ocr_filename
    if not ocr_path.exists():
        return {"text": "", "lines": []}
    
    try:
        with open(ocr_path, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
        
        # Extract text and bounding boxes from OCR data
        text_lines = []
        structured_lines = []
        
        if "recognitionResults" in ocr_data:
            for result in ocr_data["recognitionResults"]:
                if "lines" in result:
                    for line in result["lines"]:
                        if "text" in line:
                            line_info = {
                                "text": line["text"],
                                "boundingBox": line.get("boundingBox", [])
                            }
                            
                            # Extract words with their bounding boxes if available
                            words = []
                            if "words" in line:
                                for word in line["words"]:
                                    words.append({
                                        "text": word.get("text", ""),
                                        "boundingBox": word.get("boundingBox", [])
                                    })
                            line_info["words"] = words
                            
                            structured_lines.append(line_info)
                            text_lines.append(line["text"])
        
        return {
            "text": "\n".join(text_lines),
            "lines": structured_lines
        }
    except Exception as e:
        print(f"Error loading OCR file {ocr_filename}: {e}")
        return {"text": "", "lines": []}


def format_ocr_for_prompt(ocr_info: dict) -> str:
    """Format OCR data with text and bounding boxes for the prompt."""
    if not ocr_info["text"]:
        return ""
    
    ocr_formatted = "OCR text from document (with bounding boxes):\n\n"
    for idx, line_info in enumerate(ocr_info["lines"], 1):
        ocr_formatted += f"Line {idx}: \"{line_info['text']}\"\n"
        if line_info.get("boundingBox"):
            bbox = line_info["boundingBox"]
            ocr_formatted += f"  Bounding box: [{','.join(map(str, bbox))}]\n"
        
        # Include word-level bounding boxes if available
        if line_info.get("words"):
            ocr_formatted += "  Words:\n"
            for word_info in line_info["words"]:
                ocr_formatted += f"    - \"{word_info['text']}\""
                if word_info.get("boundingBox"):
                    bbox = word_info["boundingBox"]
                    ocr_formatted += f" [{','.join(map(str, bbox))}]"
                ocr_formatted += "\n"
        ocr_formatted += "\n"
    
    return ocr_formatted


# ==========================
# LOAD DATA
# ==========================
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

samples = data["data"]
total_samples = len(samples)

print(f"Starting combined generation (description + explanation) on {total_samples} samples...\n")

results = []

token_bucket = 0
minute_window_start = time.time()
request_count = 0


# ==========================
# PROCESS SAMPLES
# ==========================
for i in tqdm(range(min(total_samples, 10))):  # Limit if needed
    if request_count >= MAX_RPD:
        print(f"Daily limit reached ({MAX_RPD}). Stopping.")
        break

    # -- TPM LIMIT --
    now = time.time()
    if now - minute_window_start >= 60:
        token_bucket = 0
        minute_window_start = now

    if token_bucket + AVG_TOKENS_PER_CALL > MAX_TPM:
        wait_time = 60 - (now - minute_window_start)
        print(f"[TPM] Waiting {wait_time:.1f}s…")
        time.sleep(max(wait_time, 0))
        token_bucket = 0
        minute_window_start = time.time()

    sample = samples[i]
    image_path = IMAGE_FOLDER / Path(sample["image"]).name

    if not image_path.exists():
        print(f"[{i+1}] Missing image → skipping: {image_path}")
        continue

    question = sample.get("question", "")
    answers = sample.get("answers", [""])
    answer = answers[0] if answers else ""
    ocr_filename = sample.get("ocr", "")

    print(f"\n[{i+1}/{total_samples}] Processing: {image_path.name}")
    print(f"Question: {question[:60]}...")
    print(f"Answer: {answer}")

    # Load OCR info (text + bounding boxes)
    ocr_info = load_ocr_info(ocr_filename, OCR_FOLDER)
    ocr_available = bool(ocr_info["text"])
    
    if ocr_available:
        print(f"OCR loaded: {len(ocr_info['text'])} chars, {len(ocr_info['lines'])} lines")
    else:
        print(f"No OCR available - using vision only")

    image_part = load_image_as_part(image_path)

    # --- GENERATE COMBINED OUTPUT ---
    # Build the content list
    content_parts = [
        {"text": f"Question: {question}"},
        {"text": f"Ground truth answer: {answer}"}
    ]
    
    # Add OCR text with bounding boxes if available
    # This helps the model generate more accurate image descriptions
    if ocr_available:
        ocr_formatted = format_ocr_for_prompt(ocr_info)
        content_parts.append({"text": ocr_formatted})
        print(f"OCR data included in prompt ({len(ocr_formatted)} chars)")
    
    content_parts.extend([
        {"text": "Document image:"},
        image_part
    ])
    
    try:
        response = model.generate_content(
            content_parts,
            request_options={"timeout": 180}
        )

        raw_text = response.text.strip() if response.text else ""
        parsed = clean_json_output(raw_text)

        image_description = parsed.get("image_description", "")
        answer_explanation = parsed.get("answer_explanation", "")
        reasoning_type = parsed.get("reasoning_type", "")

        token_bucket += AVG_TOKENS_PER_CALL
        request_count += 1

        print(f"Generated: {len(image_description)} chars (desc), {len(answer_explanation)} chars (expl), reasoning: {reasoning_type}")
        if ocr_available and image_description:
            print(f"Description preview: {image_description[:100]}...")

    except Exception as e:
        print(f"Error processing sample {i}: {e}")
        image_description = ""
        answer_explanation = ""
        reasoning_type = ""

    # --- STORE RESULT ---
    results.append({
        "questionId": sample.get("questionId"),
        "question": question,
        "question_types": sample.get("question_types"),
        "image": str(image_path),
        "docId": sample.get("docId"),
        "ucsf_document_id": sample.get("ucsf_document_id"),
        "ucsf_document_page_no": sample.get("ucsf_document_page_no"),
        "ocr": ocr_filename,
        "answers": answers,
        "image_description": image_description,
        "answer_explanation": answer_explanation,
        "reasoning_type": reasoning_type
    })

    # Save intermediate progress (every sample)
    if (i + 1) % 1 == 0 or i == total_samples - 1:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
            json.dump(results, f_out, ensure_ascii=False, indent=4)
        print(f"Saved progress ({i+1}/{total_samples}).")

    time.sleep(SLEEP_BETWEEN_CALLS)


# ==========================
# DONE
# ==========================
print(f"\nCompleted {len(results)} samples.")
print(f"Results saved to: {OUTPUT_PATH}")
print(f"Total API calls: {request_count}")

