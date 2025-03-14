# scripts/prepare_small_dataset.py
import os
import argparse
import subprocess
import json
from tqdm import tqdm

def ensure_dir(directory):
    """Ensure a directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def main(args):
    """Prepare a small dataset for Text2CAD (1,000 samples)."""
    # Create output directories
    ensure_dir(args.output_dir)
    
    # Step 1: Select 1,000 samples
    print("Step 1: Selecting 1,000 samples")
    subprocess.run([
        "python", "scripts/select_samples.py",
        "--split_json", args.split_json,
        "--cad_json_dir", args.cad_json_dir,
        "--cad_vec_dir", args.cad_vec_dir,
        "--output_dir", args.output_dir,
        "--total", str(args.num_samples),
        "--seed", str(args.seed)
    ])
    
    # Step 2: Generate shape descriptions using VLM (simplified for 1,000 samples)
    print("Step 2: Generating shape descriptions (simplified for small dataset)")
    
    # Load the subset split
    with open(os.path.join(args.output_dir, "subset_split.json"), 'r') as f:
        splits = json.load(f)
    
    # Create placeholder shape descriptions
    shape_descriptions = {}
    for split_name, samples in splits.items():
        for uid in samples:
            shape_descriptions[uid] = f"A CAD model with UID {uid}"
    
    # Save placeholder shape descriptions
    ensure_dir(os.path.join(args.output_dir, "annotations"))
    with open(os.path.join(args.output_dir, "annotations", "shape_descriptions.json"), 'w') as f:
        json.dump(shape_descriptions, f, indent=2)
    
    # Step 3: Generate multi-level instructions (simplified for 1,000 samples)
    print("Step 3: Generating multi-level instructions (simplified for small dataset)")
    
    # Create placeholder annotations
    annotations = {}
    for uid, description in shape_descriptions.items():
        annotations[uid] = {
            "abstract": f"A simple CAD model.",
            "beginner": f"Create a CAD model by drawing basic shapes and extruding them.",
            "intermediate": f"Draw a sketch with precise dimensions and extrude it to create a 3D model.",
            "expert": f"Set up a coordinate system, draw a sketch with exact measurements, and extrude it with specified parameters."
        }
    
    # Save placeholder annotations
    with open(os.path.join(args.output_dir, "annotations", "text_annotations.json"), 'w') as f:
        json.dump(annotations, f, indent=2)
    
    # Also save as CSV for easier inspection
    import pandas as pd
    
    rows = []
    for uid, instructions in annotations.items():
        row = {"uid": uid}
        row.update(instructions)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    csv_file = os.path.join(args.output_dir, "annotations", "text_annotations.csv")
    df.to_csv(csv_file, index=False)
    
    # Step 4: Create processed dataset structure
    print("Step 4: Creating processed dataset structure")
    
    for split_name in splits.keys():
        split_dir = os.path.join(args.output_dir, "processed", split_name)
        ensure_dir(split_dir)
        
        # Save list of samples
        with open(os.path.join(split_dir, "samples.json"), 'w') as f:
            json.dump(splits[split_name], f, indent=2)
    
    print(f"Small dataset preparation complete! ({args.num_samples} samples)")
    print(f"You can now train the model with: python train.py --data_dir {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a small dataset for Text2CAD")
    parser.add_argument("--cad_json_dir", required=True, help="Directory containing CAD JSON files")
    parser.add_argument("--cad_vec_dir", required=True, help="Directory containing CAD vector files")
    parser.add_argument("--split_json", required=True, help="Path to train/val/test split JSON file")
    parser.add_argument("--output_dir", required=True, help="Output directory for processed data")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    main(args)