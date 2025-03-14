# model/text2cad.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from .layers.adaptive_layer import AdaptiveLayer


class Text2CAD(nn.Module):
    """
    Text2CAD transformer model for generating CAD models from text.
    """
    
    def __init__(
        self,
        bert_model="bert-base-uncased",
        d_model=768,
        nhead=8,
        num_decoder_layers=8,
        dim_feedforward=2048,
        dropout=0.1,
        vocab_size=256,
        max_seq_length=512
    ):
        """
        Initialize the Text2CAD model.
        
        Args:
            bert_model: Pre-trained BERT model name
            d_model: Dimension of the model
            nhead: Number of attention heads
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout rate
            vocab_size: Size of the vocabulary (for quantized coordinates)
            max_seq_length: Maximum sequence length
        """
        super().__init__()
        
        # Text encoder
        self.bert = BertModel.from_pretrained(bert_model)
        
        # Adaptive layer for domain-specific text features
        self.adaptive_layer = AdaptiveLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # CAD token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters for better convergence."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, text_ids, text_mask, cad_tokens=None, teacher_forcing_ratio=1.0):
        """
        Forward pass of the Text2CAD model.
        
        Args:
            text_ids: Text input ids
            text_mask: Text attention mask
            cad_tokens: CAD tokens for teacher forcing
            teacher_forcing_ratio: Ratio of teacher forcing
            
        Returns:
            Output logits for CAD tokens
        """
        # Encode text
        text_output = self.bert(text_ids, attention_mask=text_mask).last_hidden_state
        
        # Apply adaptive layer
        text_features = self.adaptive_layer(text_output.transpose(0, 1)).transpose(0, 1)
        
        # If no CAD tokens provided, use only start token
        batch_size = text_ids.size(0)
        device = text_ids.device
        
        if cad_tokens is None:
            # Start with a single start token
            cad_tokens = torch.ones(batch_size, 1, dtype=torch.long, device=device)
            
            # Generate sequence auto-regressively
            max_length = 512  # Maximum sequence length
            
            for _ in range(max_length - 1):
                # Get token embeddings
                position_ids = torch.arange(cad_tokens.size(1), device=device).unsqueeze(0)
                token_embeds = self.token_embedding(cad_tokens) + self.position_embedding(position_ids)
                
                # Run through decoder
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(cad_tokens.size(1)).to(device)
                decoder_output = self.transformer_decoder(
                    token_embeds.transpose(0, 1),
                    text_features.transpose(0, 1),
                    tgt_mask=tgt_mask
                ).transpose(0, 1)
                
                # Predict next token
                next_token_logits = self.output_projection(decoder_output[:, -1])
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                
                # Append to sequence
                cad_tokens = torch.cat([cad_tokens, next_token], dim=1)
                
                # Check if all sequences have end token
                if (cad_tokens == 2).any(dim=1).all():
                    break
            
            # Get full sequence embeddings
            position_ids = torch.arange(cad_tokens.size(1), device=device).unsqueeze(0)
            token_embeds = self.token_embedding(cad_tokens) + self.position_embedding(position_ids)
            
            # Run through decoder again
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(cad_tokens.size(1)).to(device)
            decoder_output = self.transformer_decoder(
                token_embeds.transpose(0, 1),
                text_features.transpose(0, 1),
                tgt_mask=tgt_mask
            ).transpose(0, 1)
            
            # Get full sequence logits
            logits = self.output_projection(decoder_output)
        else:
            # Teacher forcing - use ground truth tokens
            position_ids = torch.arange(cad_tokens.size(1), device=device).unsqueeze(0)
            token_embeds = self.token_embedding(cad_tokens) + self.position_embedding(position_ids)
            
            # Run through decoder
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(cad_tokens.size(1)).to(device)
            decoder_output = self.transformer_decoder(
                token_embeds.transpose(0, 1),
                text_features.transpose(0, 1),
                tgt_mask=tgt_mask
            ).transpose(0, 1)
            
            # Get logits
            logits = self.output_projection(decoder_output)
        
        return logits