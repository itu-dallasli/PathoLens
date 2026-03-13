import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from patholens.config import Config
from patholens.logger import get_logger
from patholens.sequence_model.slide_encoder import SlideEncoder

log = get_logger(__name__)

class PseudoEmbeddingsDataset(Dataset):
    """
    Temporary dataset for testing the training loop with synthetic data.
    In real usage, this should load HDF5 embeddings extracted by UNI.
    """
    def __init__(self, num_samples=100, max_seq_len=1000, embed_dim=1024, num_classes=2):
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # seq_len = torch.randint(100, self.max_seq_len, (1,)).item()
        seq_len = self.max_seq_len # Keep it fixed for simple batching in this mock loop
        x = torch.randn(seq_len, self.embed_dim)
        y = torch.randint(0, self.num_classes, (1,)).item()
        return x, y

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Mamba requires (batch, seq_len, embed_dim)
        outputs = model(inputs)
        logits = outputs.logits
        
        loss = criterion(logits, targets)
        loss.backward()
        
        # Gradient clipping for Mamba stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(logits.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100 * correct / total:.2f}%"})
        
    return total_loss / len(dataloader), correct / total

def parse_args():
    parser = argparse.ArgumentParser(description="Train the PathoLens Mamba Sequence Encoder")
    parser.add_argument("--config", type=str, default=None, help="Path to custom config.yaml")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    return parser.parse_args()

def main():
    args = parse_args()
    config = Config.load(args.config) if args.config else Config.load()
    
    device = torch.device(config.system.device)
    log.info(f"Starting training on device: {device}")
    
    # Initialize the model
    model = SlideEncoder(
        d_input=config.embedding.embedding_dim,
        d_model=config.sequence_model.d_model,
        n_layers=config.sequence_model.n_layers,
        region_size=config.sequence_model.region_size,
        n_classes=2 # e.g., Benign vs Malignant for simple pre-training
    ).to(device)
    
    # Initialize mock dataset
    dataset = PseudoEmbeddingsDataset(
        num_samples=200, 
        max_seq_len=800, 
        embed_dim=config.embedding.embedding_dim
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    best_acc = 0.0
    checkpoint_dir = Path(config.paths.checkpoints_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    log.info(f"Training for {args.epochs} epochs with {len(dataset)} samples...")
    for epoch in range(args.epochs):
        log.info(f"Epoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, dataloader, optimizer, criterion, device)
        
        log.info(f"Epoch {epoch+1} Summary - Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%")
        
        # Save best model
        if train_acc > best_acc:
            best_acc = train_acc
            save_path = checkpoint_dir / "mamba_slide_encoder_best.pt"
            torch.save(model.state_dict(), save_path)
            log.info(f"Saved best model to {save_path} (Acc: {best_acc*100:.2f}%)")

if __name__ == "__main__":
    main()
