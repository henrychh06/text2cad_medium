# scripts/explore_cad_vec.py
import os
import h5py
import argparse
import numpy as np

def explore_h5_file(file_path):
    """Explore the structure of an H5 file."""
    with h5py.File(file_path, 'r') as f:
        print(f"File: {file_path}")
        print("\nKeys:")
        for key in f.keys():
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

def main(args):
    """Explore CAD vector files."""
    # Find a sample H5 file
    if os.path.isfile(args.path):
        # Direct file path provided
        explore_h5_file(args.path)
    else:
        # Directory provided, find the first H5 file
        for root, _, files in os.walk(args.path):
            for file in files:
                if file.endswith('.h5'):
                    explore_h5_file(os.path.join(root, file))
                    if not args.all:
                        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore CAD vector files")
    parser.add_argument("path", help="Path to H5 file or directory containing H5 files")
    parser.add_argument("--all", action="store_true", help="Explore all H5 files found")
    
    args = parser.parse_args()
    main(args)