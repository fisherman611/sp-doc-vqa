import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import json
from pathlib import Path
from datetime import datetime
from qwen_vl_utils import process_vision_info
from dotenv import load_dotenv
load_dotenv()

### Login to Hugging Face Hub
from huggingface_hub import login
login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))

with open("models/qwen2_vl/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

MODEL_NAME = config["model_name"]
DATA_PATH = Path(config["data_path"])
IMAGE_FOLDER = Path(config["image_folder"])
OUTPUT_PATH = Path(config["output_path"])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
START_IDX = config["start_idx"]

with open("models/qwen2_vl/system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read().strip()

print(f"Running on device: {DEVICE}")

print(f"Loading {MODEL_NAME} and processor...")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float16, 
    device_map="auto"
).eval()

if hasattr(model, "generation_config"):
    model.generation_config.use_cache = False

processor = AutoProcessor.from_pretrained(MODEL_NAME)

print(f"Loadig data from {DATA_PATH}...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
    
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
results = []

total_samples = len(data["data"])
print(f"Starting DocVQA inference on {total_samples} samples...")

for i in range(START_IDX, min(total_samples, 10)):
    sample = data["data"][i]
    image_path = IMAGE_FOLDER / Path(sample["image"]).name
    question = sample["question"]
    
    if not image_path.exists():
        print(f"Missing image, skipping: {image_path}")
        continue

    print(f"\n[{i+1}] Processing: {image_path.name}")

    image = Image.open(image_path).convert("RGB")
    max_side = 1024  # try 768 if you still get OOM
    image.thumbnail((max_side, max_side), Image.LANCZOS)
    
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        },
    ]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(DEVICE)
    try:
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128, temperature=0.3)
            trimmed_ids = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            pred_answer = processor.batch_decode(
                trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        print("\nQuestion:", question)
        print("Image:", image_path)
        print(f"Model Answer: {pred_answer[0]}\n")
        
    except Exception as e:
        print(f"Error on sample {i}: {e}")
        pred_answer = [""]
        
    results.append(
        {
            "questionId": sample.get("questionId"),
            "question": question,
            "image": str(image_path),
            "predicted_answer": pred_answer[0],
            "ground_truth": sample.get("answers"),
            "timestamp": datetime.now().isoformat(),
        }
    )
    
    if (i + 1) % 10 == 0 or i == total_samples - 1:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
            json.dump(results, f_out, ensure_ascii=False, indent=4)
        print(f"Saved progress ({i+1}/{total_samples})")
        
print(f"\nCompleted {len(results)} samples.")
print(f"Results saved to: {OUTPUT_PATH}")