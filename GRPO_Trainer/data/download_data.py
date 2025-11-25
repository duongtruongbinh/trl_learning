from datasets import load_dataset
import os

# Define the output directory and ensure it exists
OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Starting redownload, load, and JSONL conversion...")

# Process both splits
for split_name in ['train', 'test']:
    output_json_path = os.path.join(OUTPUT_DIR, f"{split_name}.jsonl")
    
    # 1. Load the dataset split (this redownloads or uses the verified cache)
    print(f"Loading {split_name} split...")
    dataset_split = load_dataset("openai/gsm8k", 'main', split=split_name)
    
    # 2. Convert and save the split to JSONL format
    print(f"Saving {split_name} data to: {output_json_path}")
    dataset_split.to_json(output_json_path, orient='records', lines=True)
    
    print(f"✅ Successfully saved {len(dataset_split)} records.")

print("\nConversion complete.")