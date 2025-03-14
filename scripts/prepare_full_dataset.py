# scripts/prepare_full_dataset.py
import os
import argparse
import subprocess
import json
from tqdm import tqdm

from cad_processing.utils.utils import ensure_dir

def main(args):
    """Prepare the complete dataset for Text2CAD."""
    # Create output directories
    ensure_dir(args.output_dir)
    ensure_dir(os.path.join(args.output_dir, "annotations"))
    ensure_dir(os.path.join(args.output_dir, "processed"))
    
    # Step 1: Generate shape descriptions using VLM
    print("Step 1: Generating shape descriptions using VLM")
    subprocess.run([
        "python", "scripts/vlm_annotation.py",
        "--input_dir", args.cad_json_dir,
        "--output_dir", os.path.join(args.output_dir, "annotations"),
        "--split_json", args.split_json,
        "--split", args.split,
        "--max_samples", str(args.max_samples) if args.max_samples else "None",
        "--n_views", str(args.n_views),
        "--dataset", args.dataset
    ])
    
    # Step 2: Generate multi-level instructions using LLM
    print("Step 2: Generating multi-level instructions using LLM")
    subprocess.run([
        "python", "scripts/llm_annotation.py",
        "--input_dir", os.path.join(args.output_dir, "annotations"),
        "--output_dir", os.path.join(args.output_dir, "annotations"),
        "--max_samples", str(args.max_samples) if args.max_samples else "None"
    ])
    
    # Step 3: Process the dataset for training
    print("Step 3: Preparing the dataset for training")
    
    # Load annotations
    annotations_file = os.path.join(args.output_dir, "annotations", "text_annotations.json")
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    # Load split
    with open(args.split_json, 'r') as f:
        splits = json.load(f)
    
    # Create processed dataset structure
    for split_name in ["train", "test", "validation"]:
        split_dir = os.path.join(args.output_dir, "processed", split_name)
        ensure_dir(split_dir)
        
        # Get UIDs for this split
        uids = splits.get(split_name, [])
        
        # Limit to max_samples if needed
        if args.max_samples and split_name == args.split:
            uids = uids[:args.max_samples]
        
        # Create a list of processed samples
        processed_samples = []
        
        for uid in tqdm(uids, desc=f"Processing {split_name} split"):
            # Check if we have annotations for this UID
            if uid in annotations:
                # Get annotations
                annotation = annotations[uid]
                
                # Check if we have CAD vector for this UID
                cad_vec_path = os.path.join(args.cad_vec_dir, uid.split('/')[0], f"{uid.split('/')[1]}.h5")
                if os.path.exists(cad_vec_path):
                    processed_samples.append(uid)
        
        # Save list of processed samples
        with open(os.path.join(split_dir, "samples.json"), 'w') as f:
            json.dump(processed_samples, f, indent=2)
    
    print("Dataset preparation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for Text2CAD")
    parser.add_argument("--cad_json_dir", required=True, help="Directory containing CAD JSON files")
    parser.add_argument("--cad_vec_dir", required=True, help="Directory containing CAD vector files")
    parser.add_argument("--output_dir", required=True, help="Output directory for processed data")
    parser.add_argument("--split_json", required=True, help="Path to train/val/test split JSON file")
    parser.add_argument("--split", default="train", help="Dataset split to process (train, test, validation)")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument("--n_views", type=int, default=6, help="Number of views to render")
    parser.add_argument("--dataset", choices=["deepcad", "fusion360"], default="deepcad", help="Dataset type")
    
    args = parser.parse_args()
    main(args)