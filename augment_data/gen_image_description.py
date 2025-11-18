import os
import sys
import json
import time
import re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai

# ==========================
# CONFIG
# ==========================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash-lite"

IMAGE_FOLDER = Path("data/spdocvqa_images")
DATA_PATH = Path("data/spdocvqa_qas/train_v1.0_withQT.json")
OUTPUT_PATH = Path("data/augmented_data/gemini_image_descriptions.json")

# Rate limits
MAX_RPM = 15
MAX_RPD = 1000
MAX_TPM = 1_000_000
AVG_TOKENS_PER_CALL = 600
SLEEP_BETWEEN_CALLS = 60.0 / MAX_RPM

with open('augment_data/gen_image_description_prompt.txt', 'r', encoding='utf-8') as f:
    SYSTEM_PROMPT = f.read()

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(
    MODEL_NAME,
    system_instruction=SYSTEM_PROMPT.strip()
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


def clean_markdown(text: str):
    """Remove ```json fences and return raw JSON string."""
    if not text:
        return ""
    cleaned = re.sub(r"```.*?\n|```", "", text).strip()
    return cleaned

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

samples = data["data"]
total_samples = len(samples)

print(f"Starting Gemini image description on {total_samples} samples...\n")

results = []

token_bucket = 0
minute_window_start = time.time()
request_count = 0


for i in range(min(total_samples, 1000)):  # limit if needed
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

    print(f"\n[{i+1}] Describing: {image_path.name}")

    image_part = load_image_as_part(image_path)

    # --- GENERATE DESCRIPTION ---
    try:
        response = model.generate_content(
            [
                {"text": "Describe the document image below:"},
                image_part
            ],
            request_options={"timeout": 180}
        )

        description = response.text.strip() if response.text else ""
        token_bucket += AVG_TOKENS_PER_CALL
        request_count += 1

    except Exception as e:
        print(f"Error processing {i}: {e}")
        description = ""

    # --- STORE RESULT ---
    results.append({
        "questionId": sample.get("questionId"),
        "image": str(image_path),
        "description": description,
        "timestamp": datetime.now().isoformat()
    })

    # Save intermediate progress
    if (i + 1) % 1 == 0 or i == total_samples - 1:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
            json.dump(results, f_out, ensure_ascii=False, indent=4)
        print(f"Saved progress ({i+1}/{total_samples}).")

    time.sleep(SLEEP_BETWEEN_CALLS)

# ==========================
# DONE
# ==========================
print(f"\n✓ Completed {len(results)} samples.")
print(f"✓ Results saved to: {OUTPUT_PATH}")
