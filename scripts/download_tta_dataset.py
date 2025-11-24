import os
from huggingface_hub import snapshot_download

def download_tta_dataset():
    print("Downloading TTA01/AssurAI dataset (Full Repo)...")
    
    # Define local cache directory
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data_cache", "TTA01_AssurAI"))
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading to {output_dir}...")
    
    try:
        # Use snapshot_download to get all files including images
        # allow_patterns can be used if we only want specific things, but getting everything is safer for structure
        snapshot_download(
            repo_id="TTA01/AssurAI",
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False, # Download actual files
            resume_download=True
        )
        
        print(f"Dataset successfully downloaded to {output_dir}")
        print("This includes all text JSONs and image files.")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    download_tta_dataset()
