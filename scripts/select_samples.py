# scripts/select_samples.py
import os
import json
import argparse
import random
from tqdm import tqdm

def main(args):
    """Select a subset of samples from the DeepCAD dataset."""
    # Load the train/val/test split
    with open(args.split_json, 'r') as f:
        splits = json.load(f)
    
    # Count total samples
    total_samples = sum(len(split) for split in splits.values())
    print(f"Total samples in dataset: {total_samples}")
    
    # Define the number of samples for each split
    train_count = int(args.total * 0.8)  # 80% for training
    val_count = int(args.total * 0.1)   # 10% for validation
    test_count = args.total - train_count - val_count  # Remaining for testing
    
    # Randomly select samples
    train_samples = random.sample(splits["train"], min(train_count, len(splits["train"])))
    val_samples = random.sample(splits["validation"], min(val_count, len(splits["validation"])))
    test_samples = random.sample(splits["test"], min(test_count, len(splits["test"])))
    
    # Create new split file
    new_splits = {
        "train": train_samples,
        "validation": val_samples,
        "test": test_samples
    }
    
    # Save the new split file
    output_path = os.path.join(args.output_dir, "subset_split.json")
    with open(output_path, 'w') as f:
        json.dump(new_splits, f, indent=2)
    
    print(f"Selected {len(train_samples)} training, {len(val_samples)} validation, and {len(test_samples)} test samples")
    print(f"Saved to {output_path}")
    
    # Verify that the selected files exist
    print("Verifying files...")
    missing_count = 0
    
    for split_name, samples in new_splits.items():
        for uid in tqdm(samples, desc=f"Checking {split_name} files"):
            json_path = os.path.join(args.cad_json_dir, f"{uid}.json")
            vec_path = os.path.join(args.cad_vec_dir, uid.split('/')[0], f"{uid.split('/')[1]}.h5")
            
            if not os.path.exists(json_path):
                print(f"Missing JSON file: {json_path}")
                missing_count += 1
            
            if not os.path.exists(vec_path):
                print(f"Missing vector file: {vec_path}")
                missing_count += 1
    
    if missing_count > 0:
        print(f"Warning: {missing_count} files are missing")
    else:
        print("All files verified successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select a subset of samples from the DeepCAD dataset")
    parser.add_argument("--split_json", required=True, help="Path to train/val/test split JSON file")
    parser.add_argument("--cad_json_dir", required=True, help="Directory containing CAD JSON files")
    parser.add_argument("--cad_vec_dir", required=True, help="Directory containing CAD vector files")
    parser.add_argument("--output_dir", required=True, help="Output directory for the new split file")
    parser.add_argument("--total", type=int, default=1000, help="Total number of samples to select")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)