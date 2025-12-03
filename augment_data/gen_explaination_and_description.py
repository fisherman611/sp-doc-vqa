import os
import json
import time
import re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

# ==========================
# CONFIG
# ==========================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash-lite"

IMAGE_FOLDER = Path("data/spdocvqa_images")
DATA_PATH = Path("data/spdocvqa_qas/train_v1.0_withQT.json")
OUTPUT_PATH = Path("data/augmented_data/gemini_augmented_descriptions_explanations.json")

# Rate limits
MAX_RPM = 15
MAX_RPD = 1000
MAX_TPM = 1_000_000
AVG_TOKENS_PER_CALL = 800  # Higher since we're generating more content
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
        print(f"⚠ JSON parse failed: {e}")
        print(f"Raw text: {cleaned[:200]}...")
        return {}


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
for i in range(min(total_samples, 1000)):  # Limit if needed
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

    print(f"\n[{i+1}/{total_samples}] Processing: {image_path.name}")
    print(f"  Question: {question[:60]}...")
    print(f"  Answer: {answer}")

    image_part = load_image_as_part(image_path)

    # --- GENERATE COMBINED OUTPUT ---
    try:
        response = model.generate_content(
            [
                {"text": f"Question: {question}"},
                {"text": f"Ground truth answer: {answer}"},
                {"text": "Document image:"},
                image_part
            ],
            request_options={"timeout": 180}
        )

        raw_text = response.text.strip() if response.text else ""
        parsed = clean_json_output(raw_text)

        image_description = parsed.get("image_description", "")
        answer_explanation = parsed.get("answer_explanation", "")
        reasoning_type = parsed.get("reasoning_type", "")

        token_bucket += AVG_TOKENS_PER_CALL
        request_count += 1

        print(f"  ✓ Generated: {len(image_description)} chars (desc), {len(answer_explanation)} chars (expl)")

    except Exception as e:
        print(f"  ✗ Error processing sample {i}: {e}")
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
        "answers": answers,
        "image_description": image_description,
        "answer_explanation": answer_explanation,
        "reasoning_type": reasoning_type
    })

    # Save intermediate progress (every sample)
    if (i + 1) % 1 == 0 or i == total_samples - 1:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
            json.dump(results, f_out, ensure_ascii=False, indent=4)
        print(f"  💾 Saved progress ({i+1}/{total_samples}).")

    time.sleep(SLEEP_BETWEEN_CALLS)


# ==========================
# DONE
# ==========================
print(f"\n✓ Completed {len(results)} samples.")
print(f"✓ Results saved to: {OUTPUT_PATH}")
print(f"✓ Total API calls: {request_count}")

