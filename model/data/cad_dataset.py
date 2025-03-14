# model/data/cad_dataset.py
import os
import json
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from cad_processing.utils.vector_loader import load_cad_vector

class Text2CADDataset(Dataset):
    def __init__(
        self,
        cad_vec_dir,
        annotation_file,
        split_file,
        split="train",
        tokenizer_name="bert-base-uncased",
        max_seq_length=512,
        max_samples=None
    ):
        """
        Initialize Text2CAD dataset.
        
        Args:
            cad_vec_dir: Directory containing CAD vector files
            annotation_file: Path to text annotation file
            split_file: Path to train/val/test split file
            split: Dataset split (train, test, validation)
            tokenizer_name: Name of the tokenizer to use
            max_seq_length: Maximum sequence length for text
            max_samples: Maximum number of samples to use
        """
        self.cad_vec_dir = cad_vec_dir
        
        # Load train/val/test split
        with open(split_file, 'r') as f:
            splits = json.load(f)
        
        if split not in splits:
            raise ValueError(f"Split '{split}' not found in split file")
        
        # Get list of UIDs for the current split
        self.uids = splits[split]
        if max_samples is not None:
            self.uids = self.uids[:max_samples]
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_length = max_seq_length
        
        # Load or create annotations
        self.annotations = self._load_annotations(annotation_file)
        
        # Load data
        self.data = self._load_data()
    
    def _load_annotations(self, annotation_file):
        """Load text annotations from file or generate if needed."""
        if os.path.exists(annotation_file):
            # Load existing annotations
            return pd.read_csv(annotation_file)
        else:
            # Create empty annotations dataframe
            return pd.DataFrame(columns=['uid', 'abstract', 'beginner', 'intermediate', 'expert'])
    
    def _load_data(self):
        """Load CAD vectors and text data."""
        data = {}
        
        for uid in tqdm(self.uids, desc=f"Loading data"):
            try:
                # Load CAD vector
                cad_vector = load_cad_vector(self.cad_vec_dir, uid)
                
                # Get annotation if available
                annotation_row = self.annotations[self.annotations['uid'] == uid]
                
                if len(annotation_row) > 0:
                    # Use existing annotation
                    prompt_levels = ['abstract', 'beginner', 'intermediate', 'expert']
                    for level in prompt_levels:
                        if level in annotation_row.columns:
                            prompt = annotation_row[level].iloc[0]
                            
                            # Tokenize prompt
                            tokenized = self.tokenizer(
                                prompt,
                                padding="max_length",
                                truncation=True,
                                max_length=self.max_seq_length,
                                return_tensors="pt"
                            )
                            
                            # Add to data
                            data_key = f"{uid}_{level}"
                            data[data_key] = {
                                'uid': uid,
                                'cad_vector': cad_vector,
                                'prompt': prompt,
                                'input_ids': tokenized['input_ids'].squeeze(0),
                                'attention_mask': tokenized['attention_mask'].squeeze(0)
                            }
                else:
                    # No annotation available, use placeholder
                    # In a full implementation, we would generate annotations using VLM/LLM
                    placeholder_prompt = f"A CAD model with UID {uid}"
                    tokenized = self.tokenizer(
                        placeholder_prompt,
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_seq_length,
                        return_tensors="pt"
                    )
                    
                    data_key = f"{uid}_abstract"
                    data[data_key] = {
                        'uid': uid,
                        'cad_vector': cad_vector,
                        'prompt': placeholder_prompt,
                        'input_ids': tokenized['input_ids'].squeeze(0),
                        'attention_mask': tokenized['attention_mask'].squeeze(0)
                    }
            
            except Exception as e:
                print(f"Error loading data for {uid}: {e}")
        
        return data
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        key = list(self.data.keys())[idx]
        return self.data[key]