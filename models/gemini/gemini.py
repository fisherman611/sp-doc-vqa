import os
import json
import time
from datetime import datetime
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash-lite"
IMAGE_FOLDER = Path("data/spdocvqa_images")
DATA_PATH = Path("data/spdocvqa_qas/val_v1.0_withQT.json")
OUTPUT_PATH = Path("outputs/gemini_vqa_results.json")

# --- API limits (adjust to your quota) ---
MAX_RPM = 15  # requests per minute
MAX_RPD = 1000  # requests per day
MAX_TPM = 1_000_000  # tokens per minute (example; adapt to your tier)

AVG_TOKENS_PER_CALL = 500

SLEEP_BETWEEN_CALLS = 60.0 / MAX_RPM  # seconds between requests

genai.configure(api_key=GEMINI_API_KEY)

with open("models/gemini/system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

model = genai.GenerativeModel(MODEL_NAME, system_instruction=SYSTEM_PROMPT.strip())
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

results = []
start_idx = 0
token_bucket = 0
minute_window_start = time.time()

request_count = 0
total_samples = len(data["data"])
print(f"Starting DocVQA inference on {total_samples} samples...")

for i in range(start_idx, min(total_samples, 1000)):
    if request_count >= MAX_RPD:
        print(f"Daily limit ({MAX_RPD}) reached. Stopping.")
        break

    now = time.time()
    if now - minute_window_start >= 60:
        token_bucket = 0
        minute_window_start = now

    if token_bucket + AVG_TOKENS_PER_CALL > MAX_TPM:
        sleep_time = 60 - (now - minute_window_start)
        print(f"Waiting {sleep_time:.1f}s to respect TPM limit...")
        time.sleep(max(sleep_time, 0))
        token_bucket = 0
        minute_window_start = time.time()

    sample = data["data"][i]
    image_path = IMAGE_FOLDER / Path(sample["image"]).name
    question = sample["question"]

    if not image_path.exists():
        print(f"Missing image, skipping: {image_path}")
        continue

    print(f"\n[{i+1}] Processing: {image_path.name}")

    image = Image.open(image_path)

    try:
        response = model.generate_content(
            [question, image], request_options={"timeout": 180}
        )
        pred_answer = response.text.strip() if response.text else ""
        token_bucket += AVG_TOKENS_PER_CALL
        request_count += 1
    except Exception as e:
        print(f"Error on sample {i}: {e}")
        pred_answer = ""

    results.append(
        {
            "questionId": sample.get("questionId"),
            "question": question,
            "image": str(image_path),
            "predicted_answer": pred_answer,
            "ground_truth": sample.get("answers"),
            "timestamp": datetime.now().isoformat(),
        }
    )

    if (i + 1) % 10 == 0 or i == total_samples - 1:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
            json.dump(results, f_out, ensure_ascii=False, indent=4)
        print(f"Saved progress ({i+1}/{total_samples})")
    time.sleep(SLEEP_BETWEEN_CALLS)

print(f"\nCompleted {len(results)} samples.")
print(f"Results saved to: {OUTPUT_PATH}")
