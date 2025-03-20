import os
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import datasets, transforms
import pickle

# Base dataset information
DATASET_INFO = {
    'mnist': {
        'image_size': 28,
        'channels': 1,
    },
    'fashion_mnist': {
        'image_size': 28,
        'channels': 1,
    },
    'cifar10': {
        'image_size': 32,
        'channels': 3,
    }
}

def compute_dataset_stats(dataset_name, root='./data'):
    """Compute dataset mean and standard deviation."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    
    # Use ToTensor transform only for computing stats
    if dataset_name == 'mnist':
        dataset = datasets.MNIST(root=root, train=True, download=True, 
                               transform=transforms.ToTensor())
    elif dataset_name == 'fashion_mnist':
        dataset = datasets.FashionMNIST(root=root, train=True, download=True, 
                                      transform=transforms.ToTensor())
    elif dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root=root, train=True, download=True, 
                                 transform=transforms.ToTensor())
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create dataloader
    loader = DataLoader(dataset, batch_size=1000, num_workers=2, shuffle=False)
    
    # Compute mean and std
    channels = DATASET_INFO[dataset_name]['channels']
    mean = torch.zeros(channels)
    std = torch.zeros(channels)
    
    print(f"Computing statistics for {dataset_name}...")
    # Compute mean
    for images, _ in loader:
        for i in range(channels):
            mean[i] += images[:, i, :, :].mean()
    mean = mean / len(loader)
    
    # Compute std
    for images, _ in loader:
        for i in range(channels):
            std[i] += ((images[:, i, :, :] - mean[i])**2).mean()
    std = torch.sqrt(std / len(loader))
    
    print(f"Dataset {dataset_name} stats:")
    print(f"  Mean: {mean.tolist()}")
    print(f"  Std: {std.tolist()}")
    
    return mean.tolist(), std.tolist()

# Function to get dataset stats (computes them if not previously saved)
def get_dataset_stats(dataset_name, root='./data', force_compute=False):
    """Get dataset statistics, computing them if necessary."""
    import os
    import json
    
    # Path to save/load stats
    stats_dir = os.path.join(root, 'stats')
    os.makedirs(stats_dir, exist_ok=True)
    stats_file = os.path.join(stats_dir, f"{dataset_name}_stats.json")
    
    # If stats file exists and not forcing recomputation, load it
    if os.path.exists(stats_file) and not force_compute:
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        return stats['mean'], stats['std']
    
    # Otherwise compute the stats
    mean, std = compute_dataset_stats(dataset_name, root)
    
    # Save the computed stats
    with open(stats_file, 'w') as f:
        json.dump({'mean': mean, 'std': std}, f)
    
    return mean, std

def get_transform(dataset_name):
    """Get the appropriate transforms for a dataset."""
    # Get dataset stats (will compute if not already done)
    mean, std = get_dataset_stats(dataset_name)
    
    # Note: Paper specifically mentions "no random flip" for all datasets
    if dataset_name in ['mnist', 'fashion_mnist', 'cifar10']:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def load_base_dataset(dataset_name, root='./data'):
    """Load the base dataset without any splitting."""
    transform = get_transform(dataset_name)
    
    if dataset_name == 'mnist':
        train_set = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    elif dataset_name == 'fashion_mnist':
        train_set = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    elif dataset_name == 'cifar10':
        train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"Loaded {dataset_name}: {len(train_set)} training samples, {len(test_set)} test samples")
    
    return train_set, test_set

def generate_dataset_splits(dataset_name, num_splits=38, seed=42, save_dir='./data/splits'):
    """
    Generate and save the dataset splits for training.
    
    Args:
        dataset_name: Name of the dataset ('mnist', 'fashion_mnist', 'cifar10')
        num_splits: Number of different training set sizes to generate
        seed: Random seed for reproducibility
        save_dir: Directory to save the splits
        
    Returns:
        List of split information dictionaries
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load the dataset
    train_set, test_set = load_base_dataset(dataset_name)
    
    # Create directory if it doesn't exist
    save_path = os.path.join(save_dir, dataset_name)
    os.makedirs(save_path, exist_ok=True)
    
    # Define the split sizes - based on paper description
    # Starting with 2 samples and ending with the full dataset
    full_size = len(train_set)
    
    # Calculate intermediate sizes on a logarithmic scale
    # Ensure first split has exactly 2 samples and last has the full dataset
    sizes = np.unique(np.round(np.logspace(np.log10(2), np.log10(full_size), num_splits)).astype(int))
    
    # Adjust to exactly get num_splits
    if len(sizes) < num_splits:
        # Add more points near the beginning where the curve changes rapidly
        additional = num_splits - len(sizes)
        small_sizes = np.unique(np.round(np.logspace(np.log10(2), np.log10(100), additional + 1)).astype(int))
        all_sizes = np.unique(np.concatenate([small_sizes, sizes]))
        sizes = all_sizes[:num_splits]
    elif len(sizes) > num_splits:
        # Take a subset that includes the first and last
        indices = np.round(np.linspace(0, len(sizes) - 1, num_splits)).astype(int)
        sizes = sizes[indices]
    
    # Ensure the first size is exactly 2 and the last is the full dataset
    sizes[0] = 2
    sizes[-1] = full_size
    
    # Get all indices and shuffle them
    all_indices = np.arange(len(train_set))
    np.random.shuffle(all_indices)
    
    # Generate and save each split
    splits_info = []
    
    for i, size in enumerate(sizes):
        # Use the first 'size' indices from the shuffled array
        split_indices = all_indices[:size].tolist()
        
        # Save the indices for this split
        split_path = os.path.join(save_path, f"split_{i+1}.pkl")
        with open(split_path, 'wb') as f:
            pickle.dump(split_indices, f)
        
        # Record information about this split
        splits_info.append({
            'split_id': i+1,
            'size': size,
            'path': split_path
        })
        
        print(f"Created {dataset_name} split {i+1}/{num_splits} with {size} samples")
    
    # Save the overall splits information
    info_path = os.path.join(save_path, "splits_info.pkl")
    with open(info_path, 'wb') as f:
        pickle.dump(splits_info, f)
    
    return splits_info

def load_dataset_split(dataset_name, split_id, root='./data', splits_dir='./data/splits'):
    """
    Load a specific pre-generated dataset split.
    
    Args:
        dataset_name: Name of the dataset
        split_id: ID of the split to load (1-38)
        root: Root directory for the base dataset
        splits_dir: Directory where splits are saved
        
    Returns:
        train_subset: Training subset for this split
        test_set: Full test set
    """
    # Load the base dataset
    train_set, test_set = load_base_dataset(dataset_name, root)
    
    # Load the indices for this split
    split_path = os.path.join(splits_dir, dataset_name, f"split_{split_id}.pkl")
    with open(split_path, 'rb') as f:
        indices = pickle.load(f)
    
    # Create a subset using these indices
    train_subset = Subset(train_set, indices)
    
    return train_subset, test_set

class MemorizationDataset(Dataset):
    """Custom dataset wrapper that provides unique identifiers for each sample."""
    
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        # Return (image, label, unique_id)
        return x, y, idx
        
    def __len__(self):
        return len(self.dataset)

def get_dataloader(dataset, batch_size, shuffle=True):
    """Create a dataloader with appropriate configuration."""
    return DataLoader(
        MemorizationDataset(dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True
    )