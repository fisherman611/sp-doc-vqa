import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from dotenv import load_dotenv
from huggingface_hub import login

# ========= ENVIRONMENT SETUP =========
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("Warning: HUGGINGFACE_HUB_TOKEN not found. Using public access mode.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ========= LOAD MODEL & PROCESSOR =========
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
# MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
print(f"Loading model: {MODEL_NAME}")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(MODEL_NAME)

# ========= EXAMPLE IMAGE + QUESTION =========
image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
question = "How many entity in this image?"

# ========= BUILD DocVQA PROMPT =========
with open("models/qwen/system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read().strip()

messages = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_url},
            {"type": "text", "text": question},
        ],
    },
]

# ========= PREPARE INPUTS =========
text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text_prompt],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to(device)

# ========= GENERATION =========
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=64)
    trimmed_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    outputs = processor.batch_decode(
        trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

print("\nQuestion:", question)
print("Image:", image_url)
print("Model Answer:", outputs[0])
