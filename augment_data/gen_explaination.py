import os
import json
import time
import re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash-lite"

INPUT_PATH = Path("data/augmented_data/gemini_image_descriptions.json")
IMAGE_FOLDER = Path("data/spdocvqa_images")
OUTPUT_PATH = Path("outputs/gemini_answer_explanations_hops.json")
OUTPUT_PATH = Path("data/augmented_data/gemini_explaination.json")

MAX_RPM = 15
MAX_RPD = 1000
MAX_TPM = 1_000_000
AVG_TOKENS = 600
SLEEP_BETWEEN = 60.0 / MAX_RPM

with open("augment_data/gen_explaination_prompt.txt", "r", encoding="utf-8") as f:
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
        bytes_ = f.read()
    return {
        "inline_data": {
            "mime_type": mime,
            "data": bytes_
        }
    }

def clean_json_output(text: str):
    """Remove markdown fences & parse JSON object."""
    if not text:
        return {}

    cleaned = re.sub(r"```json|```", "", text).strip()
    cleaned = re.sub(r"^json\n", "", cleaned, flags=re.IGNORECASE)

    try:
        return json.loads(cleaned)
    except Exception:
        print("⚠ JSON parse failed:")
        print(cleaned)
        return {}
    
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    samples = json.load(f)

total = len(samples)
print(f"Generating explanations + hop type for {total} samples...\n")

results = []

token_bucket = 0
minute_window = time.time()
request_count = 0

for i, item in enumerate(samples):

    if request_count >= MAX_RPD:
        print("Daily limit reached.")
        break

    # ---- TPM ----
    now = time.time()
    if now - minute_window >= 60:
        token_bucket = 0
        minute_window = now

    if token_bucket + AVG_TOKENS > MAX_TPM:
        wait = 60 - (now - minute_window)
        print(f"[TPM] Waiting {wait:.1f}s…")
        time.sleep(wait)
        token_bucket = 0
        minute_window = time.time()

    qid = item.get("questionId")
    question = item.get("question", "")
    description = item.get("description", "")
    answers = item.get("answers", [""])
    answer = answers[0] if answers else ""

    img_path = Path(item["image"])
    if not img_path.exists():
        print(f"[{i+1}] Missing image → {img_path}")
        continue

    print(f"[{i+1}] Processing QID={qid}: {img_path.name}")

    image_part = load_image_as_part(img_path)

    try:
        response = model.generate_content(
            [
                {"text": f"Question: {question}"},
                {"text": f"Answer: {answer}"},
                {"text": f"Image description:\n{description}"},
                {"text": "Document image:"},
                image_part
            ],
            request_options={"timeout": 180}
        )

        raw_text = response.text.strip() if response.text else ""
        parsed = clean_json_output(raw_text)

        explanation = parsed.get("answer_explanation", "")
        hop_type = parsed.get("reasoning_type", "")

        token_bucket += AVG_TOKENS
        request_count += 1

    except Exception as e:
        print(f"Error on sample {i}: {e}")
        explanation = ""
        hop_type = ""


    item["answer_explanation"] = explanation
    item["reasoning_type"] = hop_type
    item["explanation_timestamp"] = datetime.now().isoformat()

    results.append(item)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=4)

    print(f"Saved ({i+1}/{total}).")

    time.sleep(SLEEP_BETWEEN)

print("\n✓ Completed all samples.")
print("✓ Output saved to:", OUTPUT_PATH)