"""
Training utilities for diffusion models.
Contains functions for training models on different dataset splits.
"""
import os
import time
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

# Import project modules
from src.data.datasets import load_dataset_split, DiffMemDataset
from src.models.diffusion import create_model

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def train_model(
    dataset_name, 
    split_id, 
    results_dir='./results',
    num_epochs=100,
    batch_size=None,  # If None, will be auto-determined based on dataset size
    lr=2e-5,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_every=100,
    use_wandb=True,
    wandb_project="diffusion-memorization",
    wandb_entity=None
):
    """
    Train a diffusion model on a specific dataset split.
    
    Args:
        dataset_name: Name of the dataset ('mnist', 'fashion_mnist', 'cifar10')
        split_id: ID of the split to train on (1-38)
        results_dir: Directory to save results
        num_epochs: Number of epochs to train
        batch_size: Batch size (if None, determined automatically)
        lr: Learning rate
        device: Device to train on
        save_every: Save checkpoint every N steps
        use_wandb: Whether to use Weights & Biases for tracking
        wandb_project: W&B project name
        wandb_entity: W&B entity (username or team)
    
    Returns:
        Dictionary with training info
    """
    print(f"Training {dataset_name} model on split {split_id}")
    
    # Check wandb availability
    if use_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb not available. Install with 'pip install wandb'")
        use_wandb = False
    
    # Setup results directory
    results_dir = Path(results_dir) / dataset_name / f"split_{split_id}"
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset split
    train_dataset, _ = load_dataset_split(dataset_name, split_id)
    dataset_size = len(train_dataset)
    print(f"Training set size: {dataset_size}")
    
    # Determine batch size based on dataset size if not provided
    if batch_size is None:
        if dataset_size < 10:
            batch_size = 2
        elif dataset_size < 100:
            batch_size = min(8, dataset_size)
        elif dataset_size < 1000:
            batch_size = min(16, dataset_size)
        elif dataset_size < 10000:
            batch_size = min(32, dataset_size)
        else:
            batch_size = 64
            
    # Ensure batch size isn't larger than dataset
    batch_size = min(batch_size, dataset_size)
    print(f"Using batch size: {batch_size}")
    
    # Initialize Weights & Biases
    if use_wandb:
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            config={
                "dataset": dataset_name,
                "split_id": split_id,
                "dataset_size": dataset_size,
                "epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "device": device,
            },
            name=f"{dataset_name}_split{split_id}_{dataset_size}samples"
        )
    
    # Wrap with DiffMemDataset
    train_dataset = DiffMemDataset(train_dataset, mode='diffusion')
    
    # Create dataloader
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    # Create model
    diffusion = create_model(dataset_name, device=device)
    
    # Setup optimizer
    optimizer = Adam(diffusion.parameters(), lr=lr)
    
    # Training state
    start_step = 0
    start_epoch = 0
    global_step = 0
    
    # Try to load checkpoint if exists
    latest_checkpoint = checkpoint_dir / "latest.pt"
    if latest_checkpoint.exists():
        print(f"Loading checkpoint from {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        diffusion.model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)
        print(f"Resuming from epoch {start_epoch}, step {global_step}")
    
    # Training loop
    train_losses = []
    start_time = time.time()
    
    print(f"Training for {num_epochs} epochs")
    for epoch in range(start_epoch, num_epochs):
        epoch_losses = []
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for step, batch in enumerate(pbar):
                # Move batch to device
                batch = batch.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                loss = diffusion(batch)
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                optimizer.step()
                
                # Update tracking
                loss_value = loss.item()
                epoch_losses.append(loss_value)
                pbar.set_postfix(loss=f"{loss_value:.4f}")
                
                # Log to wandb
                if use_wandb:
                    wandb.log({
                        "loss": loss_value,
                        "epoch": epoch,
                        "global_step": global_step,
                    })
                
                # Checkpointing
                if (global_step + 1) % save_every == 0:
                    checkpoint = {
                        'model': diffusion.model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'global_step': global_step,
                        'dataset_name': dataset_name,
                        'split_id': split_id,
                        'dataset_size': dataset_size
                    }
                    
                    # Save latest checkpoint
                    torch.save(checkpoint, latest_checkpoint)
                    
                    # Generate and log samples to wandb
                    if use_wandb and global_step > 0:
                        try:
                            with torch.no_grad():
                                diffusion.eval()
                                samples = diffusion.sample(batch_size=min(4, batch_size))
                                diffusion.train()
                            
                            # Log images to wandb
                            images = []
                            for i in range(len(samples)):
                                img = samples[i].cpu()
                                if img.shape[0] == 1:  # Grayscale
                                    img = img.repeat(3, 1, 1)  # Convert to RGB for wandb
                                images.append(wandb.Image(img))
                            
                            wandb.log({"generated_samples": images}, step=global_step)
                        except Exception as e:
                            print(f"Warning: Could not generate samples for wandb: {e}")
                
                global_step += 1
        
        # End of epoch
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs} complete. Average loss: {avg_epoch_loss:.6f}")
        
        # Log epoch metrics to wandb
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "epoch_loss": avg_epoch_loss,
                "epoch_time": time.time() - start_time
            })
        
        # Save checkpoint at end of epoch
        checkpoint = {
            'model': diffusion.model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,  # Save next epoch to resume from
            'global_step': global_step,
            'dataset_name': dataset_name,
            'split_id': split_id,
            'dataset_size': dataset_size,
            'train_losses': train_losses
        }
        torch.save(checkpoint, latest_checkpoint)
    
    # Training complete
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Final wandb logging
    if use_wandb:
        # Log model to wandb
        wandb.save(str(latest_checkpoint))
        
        # Log final metrics
        wandb.log({
            "training_time": training_time,
            "final_loss": train_losses[-1] if train_losses else None,
        })
        
        # Finish run
        wandb.finish()
    
    # Return training info
    return {
        'dataset_name': dataset_name,
        'split_id': split_id,
        'training_time': training_time,
        'final_loss': train_losses[-1] if train_losses else None,
        'train_losses': train_losses,
        'dataset_size': dataset_size,
        'num_epochs': num_epochs,
        'device': device
    }