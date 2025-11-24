import json
from pathlib import Path

def combine_json_files(input_paths, output_path):
    """
    Combine multiple JSON files (each containing a list of dicts)
    into one big JSON list.
    """
    combined = []

    for file_path in input_paths:
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"[WARN] File not found: {file_path}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            if isinstance(data, list):
                combined.extend(data)
            else:
                print(f"[WARN] {file_path} does not contain a list.")

    # Save final JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=4, ensure_ascii=False)

    print(f"[OK] Combined {len(input_paths)} files → {output_path}")
    print(f"[OK] Total items: {len(combined)}")


if __name__ == "__main__":
    # Example usage:
    input_files = [
        "models/multimodal_rag/multimodal_classifier/gemini/results/zeroshot_results_1000.json",
        "models/multimodal_rag/multimodal_classifier/gemini/results/zeroshot_results_1825.json",
        "models/multimodal_rag/multimodal_classifier/gemini/results/zeroshot_results_2000.json",
        "models/multimodal_rag/multimodal_classifier/gemini/results/zeroshot_results_3000.json",
        "models/multimodal_rag/multimodal_classifier/gemini/results/zeroshot_results_4000.json",
        "models/multimodal_rag/multimodal_classifier/gemini/results/zeroshot_results_5000.json",
        "models/multimodal_rag/multimodal_classifier/gemini/results/zeroshot_results_5119.json",
        "models/multimodal_rag/multimodal_classifier/gemini/results/zeroshot_results_5240.json",
    ]

    combine_json_files(
        input_paths=input_files,
        output_path="models/multimodal_rag/multimodal_classifier/gemini/results/zeroshot_results.json"
    )
