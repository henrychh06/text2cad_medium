# model/utils/metrics.py
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support


def compute_accuracy(predictions, targets, ignore_index=0):
    """
    Compute token-level accuracy.
    
    Args:
        predictions: Predicted token indices (batch_size, seq_len)
        targets: Target token indices (batch_size, seq_len)
        ignore_index: Index to ignore (usually padding)
        
    Returns:
        Accuracy as a float
    """
    # Flatten the tensors
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    # Create a mask to ignore padding tokens
    mask = (targets != ignore_index)
    
    # Count correct predictions
    correct = (predictions == targets) & mask
    
    # Calculate accuracy
    accuracy = correct.sum().float() / mask.sum().float()
    
    return accuracy.item()


def compute_primitive_f1(predictions, targets, primitive_tokens):
    """
    Compute F1 scores for primitives (lines, circles, etc.).
    
    Args:
        predictions: Predicted token indices (batch_size, seq_len)
        targets: Target token indices (batch_size, seq_len)
        primitive_tokens: Dictionary mapping primitive types to token ranges
        
    Returns:
        Dictionary of F1 scores for each primitive type
    """
    f1_scores = {}
    
    for primitive_type, token_range in primitive_tokens.items():
        # Create masks for the primitive tokens
        pred_mask = (predictions >= token_range[0]) & (predictions <= token_range[1])
        target_mask = (targets >= token_range[0]) & (targets <= token_range[1])
        
        # Calculate precision, recall, and F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            target_mask.cpu().numpy(),
            pred_mask.cpu().numpy(),
            average='binary'
        )
        
        f1_scores[primitive_type] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return f1_scores


def compute_metrics(logits, targets, ignore_index=0):
    """
    Compute various metrics for model evaluation.
    
    Args:
        logits: Model output logits (batch_size, seq_len, vocab_size)
        targets: Target token indices (batch_size, seq_len)
        ignore_index: Index to ignore (usually padding)
        
    Returns:
        Dictionary of metrics
    """
    # Get predictions
    predictions = logits.argmax(dim=-1)
    
    # Compute accuracy
    accuracy = compute_accuracy(predictions, targets, ignore_index)
    
    # Define primitive token ranges (example)
    primitive_tokens = {
        'line': (10, 50),   # Example range for line tokens
        'circle': (51, 100), # Example range for circle tokens
        'arc': (101, 150),  # Example range for arc tokens
    }
    
    # Compute F1 scores
    f1_scores = compute_primitive_f1(predictions, targets, primitive_tokens)
    
    # Combine metrics
    metrics = {
        'accuracy': accuracy,
        'f1_scores': f1_scores
    }
    
    return metrics