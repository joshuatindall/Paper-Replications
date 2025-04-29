#!/usr/bin/env python
"""
Script for training diffusion models on different dataset splits.
Used to replicate the paper's training of 38 different model sizes.
"""
import os
import sys
import argparse
import torch

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import training function
from src.training import train_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train diffusion models for the memorization paper")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["mnist", "fashion_mnist", "cifar10"],
                        help="Dataset to train on")
    parser.add_argument("--split", type=int, required=True, 
                        help="Split ID to train (1-38)")
    parser.add_argument("--epochs", type=int, default=100, 
                        help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=None, 
                        help="Batch size (default: auto)")
    parser.add_argument("--lr", type=float, default=2e-5, 
                        help="Learning rate")
    parser.add_argument("--results-dir", type=str, default="./results", 
                        help="Directory to save results")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to train on (default: auto)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="diffusion-memorization",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="Weights & Biases entity (username or team)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Train model
    train_info = train_model(
        dataset_name=args.dataset,
        split_id=args.split,
        results_dir=args.results_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity
    )
    
    print("\nTraining summary:")
    print(f"Dataset: {train_info['dataset_name']}")
    print(f"Split: {train_info['split_id']} ({train_info['dataset_size']} samples)")
    print(f"Epochs: {train_info['num_epochs']}")
    print(f"Final loss: {train_info['final_loss']:.6f}")
    print(f"Training time: {train_info['training_time']:.2f} seconds")