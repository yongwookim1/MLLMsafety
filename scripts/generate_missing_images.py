import os
import sys
import hashlib
import json
import glob
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import yaml
import argparse

import torch.multiprocessing as mp

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from models import ImageGenerator
from data import KOBBQLoader

def process_batch(rank: int, gpu_id: int, samples: list, config_path: str, output_dir: str):
    """Worker function to process a batch of samples on a specific GPU"""
    device = f"cuda:{gpu_id}"
    print(f"Worker {rank} starting on {device} with {len(samples)} samples")
    
    try:
        # Initialize generator on specific device
        generator = ImageGenerator(config_path, device=device)
        
        generated_count = 0
        skipped_count = 0
        
        for i, sample in enumerate(samples):
            sample_id = sample["sample_id"]
            context = sample["context"]
            
            if not context or not context.strip():
                skipped_count += 1
                continue

            # Generate filename
            context_hash = hashlib.md5(context.encode()).hexdigest()[:8]
            safe_sample_id = sample_id.replace("/", "_").replace("\\", "_")
            filename = f"{safe_sample_id}_{context_hash}.jpg"
            save_image_path = os.path.join(output_dir, filename)
            
            if not os.path.isabs(save_image_path):
                save_image_path = os.path.abspath(save_image_path)

            if os.path.exists(save_image_path):
                skipped_count += 1
                continue
                
            try:
                image = generator.generate(context)
                image.save(save_image_path, 'JPEG')
                generated_count += 1
                
                if (i + 1) % 10 == 0:
                    print(f"[Worker {rank}] Progress: {i + 1}/{len(samples)} - Generated: {generated_count}")
                    
            except Exception as e:
                print(f"[Worker {rank}] Error processing {sample_id}: {e}")
                continue
                
    except Exception as e:
        print(f"[Worker {rank}] Critical error: {e}")
    finally:
        print(f"[Worker {rank}] Finished. Generated: {generated_count}, Skipped: {skipped_count}")

def get_existing_sample_ids(output_dir: str) -> set:
    """Get set of sample IDs that already have images generated"""
    existing_sample_ids = set()

    # Find all jpg files in output directory
    image_pattern = os.path.join(output_dir, "*.jpg")
    image_files = glob.glob(image_pattern)

    for image_file in image_files:
        filename = os.path.basename(image_file)
        # Remove .jpg extension
        filename = filename[:-4] if filename.endswith('.jpg') else filename

        # Extract sample_id from filename (format: sample_id_hash)
        # The hash is always 8 characters at the end, preceded by underscore
        if len(filename) > 9 and filename[-9] == '_':
            # Check if the last 8 characters are hexadecimal (hash)
            hash_part = filename[-8:]
            if all(c in '0123456789abcdef' for c in hash_part):
                sample_id = filename[:-9]  # Everything before _hash
                existing_sample_ids.add(sample_id)

    return existing_sample_ids

def get_samples_to_generate(data_loader: KOBBQLoader, existing_sample_ids: set,
                          categories: list = None, bias_types: list = None,
                          max_samples_per_category: int = None) -> list:
    """Get samples that need image generation"""
    all_samples = data_loader.get_all_samples()
    samples_to_generate = []
    category_counts = {}

    for sample in all_samples:
        sample_id = sample.get("sample_id", "")
        category = sample.get("bbq_category", "")
        bias_type = sample.get("bias_type", "")

        # Skip if already exists
        if sample_id in existing_sample_ids:
            continue

        # Filter by categories if specified
        if categories:
            # Normalize categories for comparison
            target_categories = [c.lower() for c in categories]
            if category.lower() not in target_categories:
                continue

        # Filter by bias types if specified
        if bias_types and bias_type not in bias_types:
            continue
            
        # Check per-category limit
        if max_samples_per_category:
            current_count = category_counts.get(category, 0)
            if current_count >= max_samples_per_category:
                continue
            category_counts[category] = current_count + 1

        # Add context info for image generation
        sample_info = {
            "sample_id": sample_id,
            "context": sample.get("context", ""),
            "bbq_category": category,
            "bias_type": bias_type,
            "question": sample.get("question", ""),
            "label_annotation": sample.get("label_annotation", "")
        }

        samples_to_generate.append(sample_info)

    return samples_to_generate

def main():
    parser = argparse.ArgumentParser(description="Generate images for missing KoBBQ samples")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--categories", nargs="*", help="Specific categories to generate (e.g., Age, Gender_identity, Political_orientation)")
    parser.add_argument("--bias-types", nargs="*", help="Specific bias types (biased, counter_biased)")
    parser.add_argument("--max-samples", type=int, default=1000, help="Maximum number of missing samples to generate")
    parser.add_argument("--output-dir", help="Output directory (overrides config)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be generated without actually generating")
    parser.add_argument("--include-political", action="store_true", help="Include samples with political content in context")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated list of GPU IDs to use (e.g., '0,1')")
    parser.add_argument("--max-per-category", type=int, help="Maximum number of samples to generate per category")

    args = parser.parse_args()

    config_path = args.config

    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("Missing KoBBQ Image Generation")
    print("=" * 40)

    # Only load image generator if not dry run
    # Note: For single GPU, we load it here. For Multi-GPU, it's loaded in workers.
    image_generator = None
    
    data_loader = KOBBQLoader(config_path)

    gen_config = config["image_generation"]
    output_dir = args.output_dir or gen_config["output_dir"]

    os.makedirs(output_dir, exist_ok=True)

    # Find existing sample IDs
    print("Scanning existing images...")
    existing_sample_ids = get_existing_sample_ids(output_dir)
    print(f"Found {len(existing_sample_ids)} existing sample IDs")

    # Get samples that need generation
    print("Finding samples that need image generation...")
    samples_to_generate = get_samples_to_generate(
        data_loader, existing_sample_ids,
        categories=args.categories,
        bias_types=args.bias_types,
        max_samples_per_category=args.max_per_category
    )

    if args.max_samples and not args.max_per_category:
        samples_to_generate = samples_to_generate[:args.max_samples]

    print(f"Found {len(samples_to_generate)} samples that need image generation")

    if not samples_to_generate:
        print("No samples need generation. All samples already have images.")
        return

    # Show statistics
    category_counts = {}
    for sample in samples_to_generate:
        category = sample["bbq_category"]
        category_counts[category] = category_counts.get(category, 0) + 1

    print("\nMissing samples by category:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")

    if args.dry_run:
        print("\nDry run - would generate the following samples:")
        for i, sample in enumerate(samples_to_generate[:10]):
            print(f"  {sample['sample_id']} ({sample['bbq_category']})")
        if len(samples_to_generate) > 10:
            print(f"  ... and {len(samples_to_generate) - 10} more")
        return

    print(f"\nGenerating images to: {output_dir}")
    print()

    # Handle Multi-GPU generation
    gpu_ids = [int(x) for x in args.gpus.split(",")]
    num_gpus = len(gpu_ids)
    
    if num_gpus > 1 and not args.dry_run:
        print(f"Using {num_gpus} GPUs: {gpu_ids}")
        
        # Split samples into chunks
        chunk_size = len(samples_to_generate) // num_gpus + (1 if len(samples_to_generate) % num_gpus > 0 else 0)
        chunks = [samples_to_generate[i:i + chunk_size] for i in range(0, len(samples_to_generate), chunk_size)]
        
        mp.set_start_method('spawn', force=True)
        processes = []
        
        for rank, (gpu_id, chunk) in enumerate(zip(gpu_ids, chunks)):
            if not chunk:
                continue
            p = mp.Process(target=process_batch, args=(rank, gpu_id, chunk, config_path, output_dir))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
            
        print("\nMulti-GPU generation complete")
        # Show final statistics
        final_existing = get_existing_sample_ids(output_dir)
        print(f"\nFinal total images: {len(final_existing)}")
        return

    # Single GPU Logic
    if not args.dry_run:
        # Use the first specified GPU for single process
        device = f"cuda:{gpu_ids[0]}"
        print(f"Using single GPU: {device}")
        image_generator = ImageGenerator(config_path, device=device)

    for idx, sample in enumerate(tqdm(samples_to_generate, desc="Generating missing images")):
        sample_id = sample["sample_id"]
        context = sample["context"]
        category = sample["bbq_category"]

        if not context or not context.strip():
            skipped_count += 1
            continue

        # Generate filename using same format as original script
        context_hash = hashlib.md5(context.encode()).hexdigest()[:8]
        safe_sample_id = sample_id.replace("/", "_").replace("\\", "_")
        filename = f"{safe_sample_id}_{context_hash}.jpg"
        save_image_path = os.path.join(output_dir, filename)

        if not os.path.isabs(save_image_path):
            save_image_path = os.path.abspath(save_image_path)

        # Double check if file already exists
        if os.path.exists(save_image_path):
            skipped_count += 1
            continue

        try:
            if image_generator is None:
                print("Error: Image generator not initialized")
                continue

            image = image_generator.generate(context)
            image.save(save_image_path, 'JPEG')
            generated_count += 1

            if (idx + 1) % 50 == 0:
                print(f"Progress: {idx + 1}/{len(samples_to_generate)} samples processed")
                print(f"Generated: {generated_count}, Skipped: {skipped_count}")
                print(f"Current: {category} - {sample_id}")

        except Exception as e:
            print(f"\nError processing sample {sample_id}: {e}")
            continue

    print("\n" + "=" * 40)
    print("Missing image generation complete")
    print(f"Generated: {generated_count}, Skipped: {skipped_count}, Total: {len(samples_to_generate)}")
    print(f"Images saved to: {output_dir}")

    # Show final statistics
    final_existing = get_existing_sample_ids(output_dir)
    print(f"\nFinal total images: {len(final_existing)}")

if __name__ == "__main__":
    main()
