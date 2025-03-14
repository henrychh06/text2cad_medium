# model/layers/adaptive_layer.py
import torch
import torch.nn as nn


class AdaptiveLayer(nn.Module):
    """Adaptive layer to fine-tune text embeddings for CAD domain."""
    
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1):
        """
        Initialize the adaptive layer.
        
        Args:
            d_model: Dimension of the model
            nhead: Number of attention heads
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout rate
        """
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Forward pass of the adaptive layer.
        
        Args:
            src: Source tensor
            src_mask: Source mask
            src_key_padding_mask: Source key padding mask
            
        Returns:
            Transformed source tensor
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, 
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src