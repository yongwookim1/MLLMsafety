import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

def download_model(model_name: str, local_path: str):
    print(f"Downloading {model_name} to {local_path}...")
    
    os.makedirs(local_path, exist_ok=True)
    
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=local_path,
            local_dir_use_symlinks=False
        )
        print(f"✓ Successfully downloaded {model_name}")
        return True
    except Exception as e:
        print(f"✗ Error downloading {model_name}: {e}")
        return False

def main():
    config_path = "configs/config.yaml"
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    models_config = config["models"]
    
    print("=" * 60)
    print("Model Download Script")
    print("=" * 60)
    print("\nThis script will download models to local storage.")
    print("Make sure you have enough disk space and Hugging Face access.\n")
    
    success_count = 0
    total_count = 2
    
    image_model_name = models_config["image_generator"]["name"]
    image_model_path = models_config["image_generator"]["local_path"]
    
    eval_model_name = models_config["evaluator"]["name"]
    eval_model_path = models_config["evaluator"]["local_path"]
    
    print(f"\n1. Image Generator Model:")
    print(f"   Model: {image_model_name}")
    print(f"   Local Path: {image_model_path}")
    if download_model(image_model_name, image_model_path):
        success_count += 1
    
    print(f"\n2. Evaluator Model:")
    print(f"   Model: {eval_model_name}")
    print(f"   Local Path: {eval_model_path}")
    if download_model(eval_model_name, eval_model_path):
        success_count += 1
    
    print("\n" + "=" * 60)
    print(f"Download complete: {success_count}/{total_count} models downloaded")
    print("=" * 60)
    
    if success_count < total_count:
        print("\nWarning: Some models failed to download.")
        print("Please check your internet connection and Hugging Face access.")
        sys.exit(1)

if __name__ == "__main__":
    main()

