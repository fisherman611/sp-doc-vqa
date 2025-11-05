import json

# Path to your dataset file
DATA_PATH = "data/spdocvqa_qas/val_v1.0_withQT.json"

# Load the JSON file
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Count questionId entries
count = len(data["data"])

print(f"Total number of questionId entries: {count}")
