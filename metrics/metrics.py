import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from ema import *
from anls import *

# with open("results/qwen2_vl_2b_instruct_huggingface_results.json", "r", encoding="utf-8") as f:
# with open("results/gemini_vqa_results.json", "r", encoding="utf-8") as f:
# with open("results/qwen2_vl_7b_instruct_unsloth_results.json", "r", encoding="utf-8") as f:
# with open("results/qwen2_vl_7b_results.json", "r", encoding="utf-8") as f:
with open("results/qwen2_vl_2b_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

pred_map = {results[i]["questionId"]: results[i]["predicted_answer"] for i in range(len(results))}
gold_map = {results[i]["questionId"]: results[i]["ground_truth"] for i in range(len(results))}

print("Exact match accuracy: ",exact_match_accuracy(pred_map=pred_map, gold_map=gold_map))
print("Average Normalized Levenhstein Similarity: ", average_normalized_levenshtein_similarity(pred_map=pred_map, gold_map=gold_map))