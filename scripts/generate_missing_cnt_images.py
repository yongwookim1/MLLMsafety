import os
import sys
import hashlib
import json
from pathlib import Path
from tqdm import tqdm
import yaml

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from models import ImageGenerator
from data import KOBBQLoader

def main():
    config_path = "configs/config.yaml"
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    print("Loading configuration...")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Initialize components
    print("Initializing loader and generator...")
    image_generator = ImageGenerator(config_path)
    data_loader = KOBBQLoader(config_path)

    # Setup output directory
    output_dir = config["image_generation"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # 1. Fetch samples to balance bsd/cnt
    print("Fetching samples to balance bsd/cnt counts...")
    
    # Get all samples first
    all_samples = data_loader.get_all_unique_samples(max_samples=None)
    
    # Group by category and type
    from collections import defaultdict
    cat_counts = defaultdict(lambda: {"bsd": 0, "cnt": [], "bsd_ids": set()})
    
    for s in all_samples:
        cat = s.get("bbq_category", "Unknown")
        btype = s.get("bias_type")
        if btype == "biased":
            cat_counts[cat]["bsd"] += 1
            cat_counts[cat]["bsd_ids"].add(s["sample_id"])
        elif btype == "counter_biased":
            cat_counts[cat]["cnt"].append(s)
            
    # Select cnt samples matching bsd count
    target_cnt_samples = []
    for cat, data in cat_counts.items():
        bsd_count = data["bsd"]
        cnt_list = data["cnt"]
        
        # If we have more cnt than bsd, truncate to bsd count
        # If we have less, take all (or handled by list slicing safely)
        take_count = min(len(cnt_list), bsd_count)
        selected = cnt_list[:take_count]
        
        target_cnt_samples.extend(selected)
        print(f"Category '{cat}': Found {bsd_count} bsd, selecting {len(selected)}/{len(cnt_list)} cnt samples")
        
    cnt_contexts = target_cnt_samples
    print(f"Total counter-biased samples to process: {len(cnt_contexts)}")

    # 2. Load existing images map to skip already generated ones
    mapping_file = config["outputs"]["image_context_mapping"]
    existing_contexts = set()
    if os.path.exists(mapping_file):
        with open(mapping_file, "r", encoding="utf-8") as f:
            mapping = json.load(f)
            for ctx, path in mapping.get("context_to_image", {}).items():
                if os.path.exists(path):
                    existing_contexts.add(ctx)

    # 3. Generate missing images
    generated_count = 0
    skipped_count = 0
    
    print(f"Starting generation for missing {len(cnt_contexts)} samples...")
    
    with tqdm(cnt_contexts, desc="Generating missing cnt images") as pbar:
        for item in pbar:
            context = item["context"]
            sample_id = item["sample_id"]
            
            # Skip if already exists (context level)
            if context in existing_contexts:
                skipped_count += 1
                continue
                
            # Construct path
            context_hash = hashlib.md5(context.encode()).hexdigest()[:8]
            safe_sample_id = sample_id.replace("/", "_").replace("\\", "_")
            filename = f"{safe_sample_id}_{context_hash}.jpg"
            save_path = os.path.join(output_dir, filename)
            
            if not os.path.isabs(save_path):
                save_path = os.path.abspath(save_path)
            
            # Skip if file exists (file level)
            if os.path.exists(save_path):
                skipped_count += 1
                continue
            
            try:
                # Generate
                image = image_generator.generate(context)
                image.save(save_path, 'JPEG')
                generated_count += 1
                
                pbar.set_postfix({
                    'gen': generated_count, 
                    'skip': skipped_count
                })
            except Exception as e:
                print(f"Error: {e}")
                continue

    print("\nDone!")
    print(f"Generated: {generated_count}")
    print(f"Skipped: {skipped_count}")

if __name__ == "__main__":
    main()

