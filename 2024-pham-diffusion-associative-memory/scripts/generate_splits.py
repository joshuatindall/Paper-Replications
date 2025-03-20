import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.datasets import generate_dataset_splits

def main():
    parser = argparse.ArgumentParser(description='Generate dataset splits for diffusion training')
    parser.add_argument('--datasets', nargs='+', default=['mnist', 'fashion_mnist', 'cifar10'],
                        help='Datasets to generate splits for')
    parser.add_argument('--num-splits', type=int, default=38,
                        help='Number of splits to generate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--save-dir', type=str, default='./data/splits',
                        help='Directory to save splits')
    
    args = parser.parse_args()
    
    for dataset_name in args.datasets:
        print(f"Generating splits for {dataset_name}...")
        splits_info = generate_dataset_splits(
            dataset_name, 
            num_splits=args.num_splits,
            seed=args.seed,
            save_dir=args.save_dir
        )
        
        # Print summary
        print(f"\nGenerated {len(splits_info)} splits for {dataset_name}:")
        print(f"  Smallest split: {splits_info[0]['size']} samples")
        print(f"  Largest split: {splits_info[-1]['size']} samples")
        print(f"  Splits saved to: {os.path.dirname(splits_info[0]['path'])}\n")

if __name__ == "__main__":
    main()