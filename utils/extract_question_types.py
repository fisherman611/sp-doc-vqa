import json
from collections import Counter

with open("data/spdocvqa_qas/train_v1.0_withQT.json", "r", encoding="utf-8") as f:
    data = json.load(f)

question_type_counter = Counter()

for item in data["data"]:
    qts = item.get("question_types", [])
    question_type_counter.update(qts)

for qt, count in question_type_counter.items():
    print(f"{qt}: {count}")
