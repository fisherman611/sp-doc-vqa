import json
from pathlib import Path


def add_ocr_to_json(
    input_json_path: str,
    output_json_path: str,
    ocr_folder: str = "data/spdocvqa_ocr"
) -> None:
    """
    Add OCR filename to each sample in the JSON file.
    
    Args:
        input_json_path: Path to input JSON file (train or val)
        output_json_path: Path to output JSON file with OCR filename added
        ocr_folder: Path to folder containing OCR JSON files (for validation)
    """
    print(f"Loading data from: {input_json_path}")
    
    # Load the main JSON file
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data["data"]
    total_samples = len(samples)
    ocr_folder_path = Path(ocr_folder)
    
    print(f"Processing {total_samples} samples...")
    
    added_count = 0
    missing_count = 0
    missing_files = []
    
    # Process each sample
    for i, sample in enumerate(samples):
        # Construct OCR filename from ucsf_document_id and page number
        ucsf_doc_id = sample.get("ucsf_document_id", "")
        page_no = sample.get("ucsf_document_page_no", "")
        
        if not ucsf_doc_id or not page_no:
            print(f"  [{i+1}/{total_samples}] Warning: Missing ucsf_document_id or page_no for questionId {sample.get('questionId')}")
            sample["ocr"] = None
            missing_count += 1
            continue
        
        # Construct OCR filename
        ocr_filename = f"{ucsf_doc_id}_{page_no}.json"
        ocr_file_path = ocr_folder_path / ocr_filename
        
        # Add OCR filename (just the filename, not the content)
        if ocr_file_path.exists():
            sample["ocr"] = ocr_filename
            added_count += 1
            
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i+1}/{total_samples} samples...")
        else:
            sample["ocr"] = None
            missing_count += 1
            if len(missing_files) < 10:  # Only show first 10 missing files
                missing_files.append(ocr_filename)
    
    # Save the updated JSON
    print(f"\nSaving updated data to: {output_json_path}")
    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    # Print summary
    print("\n" + "="*60)
    print("Summary:")
    print(f"  Total samples: {total_samples}")
    print(f"  OCR filenames added: {added_count}")
    print(f"  OCR files missing: {missing_count}")
    
    if missing_files:
        print(f"\n  First missing OCR files (showing {len(missing_files)}):")
        for fname in missing_files:
            print(f"    - {fname}")
    
    print("="*60)
    print(f"✓ Done! Output saved to: {output_json_path}")


if __name__ == "__main__":
    # Add OCR filename to training data
    print("\n" + "="*60)
    print("Adding OCR filename to TRAINING data")
    print("="*60)
    add_ocr_to_json(
        input_json_path="data/spdocvqa_qas/train_v1.0_withQT.json",
        output_json_path="data/spdocvqa_qas/train_v1.0_withQT_ocr.json",
        ocr_folder="data/spdocvqa_ocr"
    )
    
    print("\n\n" + "="*60)
    print("Adding OCR filename to VALIDATION data")
    print("="*60)
    # Add OCR filename to validation data
    add_ocr_to_json(
        input_json_path="data/spdocvqa_qas/val_v1.0_withQT.json",
        output_json_path="data/spdocvqa_qas/val_v1.0_withQT_ocr.json",
        ocr_folder="data/spdocvqa_ocr"
    )

