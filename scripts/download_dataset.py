import os
import sys
from pathlib import Path
from datasets import load_dataset
import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

def main():
    config_path = "configs/config.yaml"
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    dataset_config = config["dataset"]
    dataset_name = dataset_config["name"]
    dataset_split = dataset_config["split"]
    cache_dir = os.path.abspath(dataset_config.get("local_cache_dir", "./data_cache"))
    dataset_path = os.path.join(cache_dir, dataset_name.replace("/", "_"), dataset_split)
    
    print("=" * 60)
    print("Dataset Download Script")
    print("=" * 60)
    print(f"\nDataset: {dataset_name}")
    print(f"Split: {dataset_split}")
    print(f"Save Directory: {dataset_path}")
    print("\nThis script will download the dataset and save to disk.")
    print("Make sure you have internet connection and Hugging Face access.\n")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        print(f"Downloading {dataset_name}...")
        dataset = load_dataset(
            dataset_name,
            split=dataset_split
        )
        print(f"✓ Successfully downloaded dataset")
        print(f"✓ Dataset size: {len(dataset)} samples")
        
        print(f"\nSaving dataset to disk...")
        dataset.save_to_disk(dataset_path)
        print(f"✓ Successfully saved dataset to {dataset_path}")
        
        print("\n" + "=" * 60)
        print("Download complete! You can now use the dataset offline.")
        print(f"Dataset is saved at: {dataset_path}")
        print("=" * 60)
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        print("Please check your internet connection and Hugging Face access.")
        sys.exit(1)

if __name__ == "__main__":
    main()

