import os
import sys
import json
import google.generativeai as genai
from pathlib import Path
from PIL import Image
import time
import re
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash-lite"
IMAGE_FOLDER = Path("data/spdocvqa_images")
DATA_PATH = Path("data/spdocvqa_qas/val_v1.0_withQT.json")
OUTPUT_PATH = Path("models/multimodal_rag/multimodal_classifier/gemini/results/zeroshot_results_1000.json")
# OUTPUT_PATH = Path("models/multimodal_rag/multimodal_classifier/gemini/results/zeroshot_results_2000.json")
# OUTPUT_PATH = Path("models/multimodal_rag/multimodal_classifier/gemini/results/zeroshot_results_3000.json")
# OUTPUT_PATH = Path("models/multimodal_rag/multimodal_classifier/gemini/results/zeroshot_results_4000.json")
# OUTPUT_PATH = Path("models/multimodal_rag/multimodal_classifier/gemini/results/zeroshot_results_5000.json")
# OUTPUT_PATH = Path("models/multimodal_rag/multimodal_classifier/gemini/results/zeroshot_results_remainder.json")


# Rate limits (adjust to your tier)
MAX_RPM = 15
MAX_RPD = 1000
MAX_TPM = 1_000_000    # tokens per minute
AVG_TOKENS_PER_CALL = 600
SLEEP_BETWEEN_CALLS = 60.0 / MAX_RPM

with open("models/multimodal_rag/multimodal_classifier/gemini/system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(
    MODEL_NAME,
    system_instruction=SYSTEM_PROMPT
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_image_as_part(path: Path):
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        img_bytes = f.read()
    return {
        "inline_data": {
            "mime_type": mime,
            "data": img_bytes
        }
    }

def parse_markdown_json(md_json):
    cleaned = re.sub(r"```.*?\n|```", "", md_json).strip()
    return json.loads(cleaned)


with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

samples = data["data"]
total_samples = len(samples)
print(f"Starting SP-DocVQA classification on {total_samples} samples...\n")

results = []
start_idx = 0

token_bucket = 0
minute_window_start = time.time()
request_count = 0

for i in range(start_idx, 1000):  # limit if needed
    if request_count >= MAX_RPD:
        print(f"Daily limit reached ({MAX_RPD} requests). Stopping.")
        break

    # ---- TPM Bucket Handling ---- #
    now = time.time()
    if now - minute_window_start >= 60:
        token_bucket = 0
        minute_window_start = now

    if token_bucket + AVG_TOKENS_PER_CALL > MAX_TPM:
        wait_time = 60 - (now - minute_window_start)
        print(f"[TPM] Waiting {wait_time:.1f}s...")
        time.sleep(max(wait_time, 0))
        token_bucket = 0
        minute_window_start = time.time()

    sample = samples[i]
    question = sample["question"]
    image_path = IMAGE_FOLDER / Path(sample["image"]).name

    if not image_path.exists():
        print(f"[{i+1}] Missing image → skipping: {image_path}")
        continue

    print(f"\n[{i+1}] Classifying: {image_path.name}")

    # image = Image.open(image_path)
    image_part = load_image_as_part(image_path)

    # ------------ GENERATE CLASSIFICATION ------------ #
    try:
        response = model.generate_content(
            [
                {"text": f"Question: {question}"},
                {"text": "Document image:"},
                image_part
            ],
            request_options={"timeout": 180}
        )

        pred_json = response.text.strip() if response.text else "{}"
        token_bucket += AVG_TOKENS_PER_CALL
        request_count += 1

    except Exception as e:
        print(f"Error processing sample {i}: {e}")
        pred_json = "{}"

    # Save result entry
    results.append(
        {
            "questionId": sample.get("questionId"),
            "question": question,
            "image": str(image_path),
            "predicted_json": parse_markdown_json(pred_json),
            "predicted_labels": parse_markdown_json(pred_json).get("predicted_labels", []),
            "ground_truth": sample.get("question_types"),
            "timestamp": datetime.now().isoformat(),
        }
    )

    # Save intermediate progress
    if (i + 1) % 1 == 0 or i == total_samples - 1:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
            json.dump(results, f_out, ensure_ascii=False, indent=4)
        print(f"Progress saved ({i+1}/{total_samples}).")

    time.sleep(SLEEP_BETWEEN_CALLS)

print(f"\n✓ Completed {len(results)} samples.")
print(f"✓ Results saved to: {OUTPUT_PATH}")