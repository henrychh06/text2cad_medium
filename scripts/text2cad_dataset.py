# model/data/text2cad_dataset.py
import os
import json
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoTokenizer

class Text2CADDataset(Dataset):
    """Dataset for Text2CAD model."""
    
    def __init__(
        self,
        data_dir,
        cad_vec_dir,
        split="train",
        text_level="expert",
        tokenizer_name="bert-base-uncased",
        max_seq_length=512
    ):
        """
        Initialize the Text2CAD dataset.
        
        Args:
            data_dir: Directory containing processed data
            cad_vec_dir: Directory containing CAD vector files
            split: Dataset split (train, test, validation)
            text_level: Text description level (abstract, beginner, intermediate, expert)
            tokenizer_name: Name of the tokenizer to use
            max_seq_length: Maximum sequence length for text
        """
        self.data_dir = data_dir
        self.cad_vec_dir = cad_vec_dir
        self.split = split
        self.text_level = text_level
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_length = max_seq_length
        
        # Load annotations
        annotations_file = os.path.join(data_dir, "annotations", "text_annotations.json")
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Load list of samples for this split
        samples_file = os.path.join(data_dir, "processed", split, "samples.json")
        with open(samples_file, 'r') as f:
            self.samples = json.load(f)
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        uid = self.samples[idx]
        
        # Load CAD vector
        cad_vec = self._load_cad_vector(uid)
        
        # Get text annotation
        text = self.annotations[uid][self.text_level]
        
        # Tokenize text
        tokenized = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        
        # Create sample
        sample = {
            "uid": uid,
            "cad_vec": cad_vec,
            "text": text,
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0)
        }
        
        return sample
    
    def _load_cad_vector(self, uid):
        """Load CAD vector from H5 file."""
        folder_id, file_id = uid.split('/')
        h5_path = os.path.join(self.cad_vec_dir, folder_id, f"{file_id}.h5")
        
        with h5py.File(h5_path, 'r') as f:
            # Assume the CAD vector is stored in a dataset called 'vec'
            # This may need to be adjusted based on the actual structure of the H5 files
            cad_vec = torch.tensor(f["vec"][()])
        
        return cad_vec

def get_dataloaders(
    data_dir,
    cad_vec_dir,
    batch_size=16,
    text_level="expert",
    tokenizer_name="bert-base-uncased",
    max_seq_length=512,
    num_workers=4
):
    """Get dataloaders for training, validation, and testing."""
    # Create datasets
    train_dataset = Text2CADDataset(
        data_dir=data_dir,
        cad_vec_dir=cad_vec_dir,
        split="train",
        text_level=text_level,
        tokenizer_name=tokenizer_name,
        max_seq_length=max_seq_length
    )
    
    val_dataset = Text2CADDataset(
        data_dir=data_dir,
        cad_vec_dir=cad_vec_dir,
        split="validation",
        text_level=text_level,
        tokenizer_name=tokenizer_name,
        max_seq_length=max_seq_length
    )
    
    test_dataset = Text2CADDataset(
        data_dir=data_dir,
        cad_vec_dir=cad_vec_dir,
        split="test",
        text_level=text_level,
        tokenizer_name=tokenizer_name,
        max_seq_length=max_seq_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader