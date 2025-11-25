#!/usr/bin/env python3
"""
Manual TTA Dataset Downloader for Firewall Environments

This script downloads the TTA01/AssurAI dataset manually without using huggingface_hub,
which is useful when HuggingFace connections are blocked by firewall.

Run this on a machine with internet access, then transfer the downloaded files to your server.
"""

import os
import requests
import json
import argparse
from tqdm import tqdm

def download_file(url, local_path, chunk_size=8192):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(local_path, 'wb') as file, tqdm(
        desc=os.path.basename(local_path),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            file.write(chunk)
            bar.update(len(chunk))

def download_tta_dataset(output_dir="./data_cache/TTA01_AssurAI"):
    """
    Manually download TTA01/AssurAI dataset files including images

    This function downloads the dataset files directly from HuggingFace
    without using the huggingface_hub library.
    """

    os.makedirs(output_dir, exist_ok=True)

    base_url = "https://huggingface.co/datasets/TTA01/AssurAI/resolve/main"

    # Core dataset files
    core_files = [
        "data-00000-of-00001.arrow",  # Main dataset file
        "dataset_info.json",          # Dataset metadata
        "state.json"                  # Dataset state
    ]

    print(f"Downloading TTA01/AssurAI dataset to {output_dir}")
    print("=" * 50)

    # Download core dataset files
    for filename in core_files:
        url = f"{base_url}/{filename}"
        local_path = os.path.join(output_dir, filename)

        if os.path.exists(local_path):
            print(f"Skipping {filename} (already exists)")
            continue

        try:
            print(f"Downloading {filename}...")
            download_file(url, local_path)
            print(f"✓ Downloaded {filename}")
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")

    # Download image files from all risk categories
    print("\nDownloading image files...")
    image_extensions = ['.png', '.jpg', '.jpeg', '.webp']

    # Risk categories that contain images (based on dataset structure)
    risk_categories = [f"Risk_{i:02d}" for i in range(1, 16)]

    total_images = 0
    downloaded_images = 0

    for risk_cat in risk_categories:
        # Check if this risk category has images
        try:
            tree_url = f"https://huggingface.co/api/datasets/TTA01/AssurAI/tree/main/{risk_cat}/image?recursive=true"
            response = requests.get(tree_url)
            if response.status_code == 200:
                image_data = response.json()
                for item in image_data:
                    if isinstance(item, dict) and 'path' in item:
                        filepath = item['path']
                        if any(filepath.lower().endswith(ext) for ext in image_extensions):
                            total_images += 1
                            local_path = os.path.join(output_dir, filepath)

                            # Create subdirectories if needed
                            os.makedirs(os.path.dirname(local_path), exist_ok=True)

                            if os.path.exists(local_path):
                                continue

                            try:
                                url = f"{base_url}/{filepath}"
                                download_file(url, local_path)
                                downloaded_images += 1
                                if downloaded_images % 10 == 0:
                                    print(f"Downloaded {downloaded_images}/{total_images} images...")
                            except Exception as e:
                                print(f"✗ Failed to download {filepath}: {e}")

        except Exception as e:
            # This risk category might not have images, continue
            continue

    print(f"✓ Downloaded {downloaded_images} image files")

    print("\n" + "=" * 50)
    print("Download complete!")
    print(f"Files saved to: {output_dir}")
    print(f"Core files: {len(core_files)}")
    print(f"Image files: {downloaded_images}")
    print("\nNext steps:")
    print("1. Transfer the entire TTA01_AssurAI folder to your server")
    print("2. Run: python scripts/prepare_tta_data.py")
    print("3. Run: python scripts/run_tta_evaluation.py")

def verify_download(output_dir="./data_cache/TTA01_AssurAI"):
    """Verify that the downloaded files are valid"""
    required_files = [
        "data-00000-of-00001.arrow",
        "dataset_info.json"
    ]

    print("Verifying downloaded files...")

    all_present = True

    # Check core files
    for filename in required_files:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"✓ {filename} ({size} bytes)")
        else:
            print(f"✗ {filename} (missing)")
            all_present = False

    # Check for image files
    image_count = 0
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                image_count += 1

    if image_count > 0:
        print(f"✓ Found {image_count} image files")
    else:
        print("⚠️  No image files found (this might be normal if dataset has no images)")

    return all_present

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manually download TTA01/AssurAI dataset")
    parser.add_argument("--output-dir", default="./data_cache/TTA01_AssurAI",
                       help="Output directory for downloaded files")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify existing download without downloading")

    args = parser.parse_args()

    if args.verify_only:
        success = verify_download(args.output_dir)
        if success:
            print("\n✓ All required files are present!")
        else:
            print("\n✗ Some files are missing. Run without --verify-only to download.")
    else:
        try:
            download_tta_dataset(args.output_dir)
            print("\nVerifying download...")
            verify_download(args.output_dir)
        except KeyboardInterrupt:
            print("\nDownload interrupted by user.")
        except Exception as e:
            print(f"\nDownload failed: {e}")
