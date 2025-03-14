# train_small.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from model.data.text2cad_dataset import Text2CADDataset
from model.text2cad import Text2CAD

def main(args):
    """Train Text2CAD model on a small dataset."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = Text2CADDataset(
        data_dir=args.data_dir,
        cad_vec_dir=args.cad_vec_dir,
        split="train",
        text_level=args.text_level,
        tokenizer_name=args.tokenizer,
        max_seq_length=args.max_seq_length
    )
    
    val_dataset = Text2CADDataset(
        data_dir=args.data_dir,
        cad_vec_dir=args.cad_vec_dir,
        split="validation",
        text_level=args.text_level,
        tokenizer_name=args.tokenizer,
        max_seq_length=args.max_seq_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = Text2CAD(
        bert_model=args.tokenizer,
        d_model=args.d_model,
        nhead=args.nhead,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        vocab_size=args.vocab_size,
        max_seq_length=args.max_seq_length
    ).to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Create loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} (Train)"):
            # Get batch data
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cad_vec = batch["cad_vec"].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, cad_vec)
            
            # Calculate loss
            loss = criterion(outputs.view(-1, args.vocab_size), cad_vec.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            
            # Update loss
            train_loss += loss.item()
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} (Val)"):
                # Get batch data
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                cad_vec = batch["cad_vec"].to(device)
                
                # Forward pass
                outputs = model(input_ids, attention_mask, cad_vec)
                
                # Calculate loss
                loss = criterion(outputs.view(-1, args.vocab_size), cad_vec.view(-1))
                
                # Update loss
                val_loss += loss.item()
        
        # Calculate average validation loss
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Print losses
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        # Save model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Create output directory
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Save model
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            print(f"  Model saved to {os.path.join(args.output_dir, 'best_model.pt')}")
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "loss_curve.png"))
    
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Text2CAD model on a small dataset")
    parser.add_argument("--data_dir", required=True, help="Directory containing processed data")
    parser.add_argument("--cad_vec_dir", required=True, help="Directory containing CAD vector files")
    parser.add_argument("--output_dir", required=True, help="Output directory for saved models")
    parser.add_argument("--text_level", choices=["abstract", "beginner", "intermediate", "expert"], default="expert", help="Text description level")
    parser.add_argument("--tokenizer", default="bert-base-uncased", help="Tokenizer name")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length for text")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_decoder_layers", type=int, default=6, help="Number of decoder layers")
    parser.add_argument("--dim_feedforward", type=int, default=2048, help="Feedforward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--vocab_size", type=int, default=256, help="Vocabulary size")
    parser.add_argument("--cpu", action="store_true", help="Force using CPU instead of GPU")
    
    args = parser.parse_args()
    main(args)