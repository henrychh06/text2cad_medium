# scripts/explore_h5_files.py
import os
import h5py
import argparse
import numpy as np
from tqdm import tqdm

def explore_h5_file(file_path):
    """Explore the structure of an H5 file."""
    with h5py.File(file_path, 'r') as f:
        print(f"File: {file_path}")
        print("\nKeys:")
        for key in f.keys():
            try:
                data = f[key][()]
                print(f"  {key}: Shape={data.shape}, Type={data.dtype}")
                
                # Print sample data if it's small enough
                if np.prod(data.shape) < 10:
                    print(f"    Data: {data}")
                else:
                    if len(data.shape) == 1:
                        print(f"    First few elements: {data[:5]}")
                    else:
                        print(f"    First element: {data[0]}")
            except Exception as e:
                print(f"  {key}: Error - {e}")
        
        # Check for nested groups and datasets
        print("\nGroups and nested datasets:")
        
        def print_group(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"  Group: {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name}, Shape={obj.shape}, Type={obj.dtype}")
        
        f.visititems(print_group)

def main(args):
    """Explore H5 files in the cad_vec directory."""
    # Get all H5 files
    all_h5_files = []
    
    if os.path.isfile(args.path) and args.path.endswith('.h5'):
        all_h5_files = [args.path]
    else:
        for root, _, files in os.walk(args.path):
            for file in files:
                if file.endswith('.h5'):
                    all_h5_files.append(os.path.join(root, file))
    
    # Limit to max_files
    if args.max_files and len(all_h5_files) > args.max_files:
        all_h5_files = all_h5_files[:args.max_files]
    
    print(f"Found {len(all_h5_files)} H5 files")
    
    # Explore files
    for file_path in tqdm(all_h5_files, desc="Exploring H5 files"):
        try:
            explore_h5_file(file_path)
            print("\n" + "-" * 80 + "\n")
        except Exception as e:
            print(f"Error exploring {file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore H5 files in the cad_vec directory")
    parser.add_argument("path", help="Path to H5 file or directory containing H5 files")
    parser.add_argument("--max_files", type=int, default=5, help="Maximum number of files to explore")
    
    args = parser.parse_args()
    main(args)