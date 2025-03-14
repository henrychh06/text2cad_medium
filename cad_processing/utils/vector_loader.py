# cad_processing/utils/vector_loader.py
import os
import h5py
import numpy as np
import torch

def load_cad_vector(cad_vec_dir, uid):
    """
    Load CAD vector from H5 file.
    
    Args:
        cad_vec_dir: Base directory containing CAD vector files
        uid: UID of the CAD model (e.g., "0000/00000007")
        
    Returns:
        Dictionary containing CAD vector data
    """
    # Split UID into directory and file components
    dir_id, file_id = uid.split('/')
    
    # Construct file path
    file_path = os.path.join(cad_vec_dir, dir_id, f"{file_id}.h5")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CAD vector file not found: {file_path}")
    
    # Load H5 file
    with h5py.File(file_path, 'r') as f:
        # Extract vector data
        # Note: This assumes a specific structure for the H5 file
        # We'll need to adjust based on the actual structure
        data = {}
        
        # Try to extract common datasets
        for key in f.keys():
            data[key] = torch.tensor(f[key][()])
        
        return data