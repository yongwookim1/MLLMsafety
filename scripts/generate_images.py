import os
import sys
import hashlib
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from models import ImageGenerator
from data import KOBBQLoader

def main():
    config_path = "configs/config.yaml"
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("KOBBQ Image Generation Pipeline")
    print("=" * 60)
    
    image_generator = ImageGenerator(config_path)
    data_loader = KOBBQLoader(config_path)
    
    gen_config = config["image_generation"]
    output_dir = gen_config["output_dir"]
    max_contexts = gen_config["max_contexts"]
    output_config = config["outputs"]
    mapping_file = output_config["image_context_mapping"]
    
    os.makedirs(output_dir, exist_ok=True)
    
    existing_contexts = set()
    if os.path.exists(mapping_file):
        print(f"Loading existing image-context mapping from {mapping_file}...")
        try:
            with open(mapping_file, "r", encoding="utf-8") as f:
                mapping = json.load(f)
                context_to_image = mapping.get("context_to_image", {})
                for context, image_path in context_to_image.items():
                    if os.path.exists(image_path):
                        existing_contexts.add(context)
                print(f"Found {len(existing_contexts)} existing contexts with images")
        except Exception as e:
            print(f"Warning: Failed to load mapping file: {e}")
    
    print(f"Loading unique contexts (max: {max_contexts})...")
    contexts = data_loader.get_unique_contexts(max_contexts=max_contexts)
    
    print(f"Total contexts to generate: {len(contexts)}")
    print(f"Output directory: {output_dir}\n")
    
    generated_count = 0
    skipped_count = 0
    
    for idx, item in enumerate(tqdm(contexts, desc="Generating images")):
        context = item["context"]
        sample_id = item["sample_id"]
        
        if context in existing_contexts:
            skipped_count += 1
            continue
        
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
            image = image_generator.generate(context)
            image.save(save_image_path, 'JPEG')
            generated_count += 1
            
            if (idx + 1) % 100 == 0:
                print(f"\nProgress: {idx + 1}/{len(contexts)} contexts processed")
                print(f"  Generated: {generated_count}, Skipped: {skipped_count}")
        
        except Exception as e:
            print(f"\nError processing context (sample_id: {sample_id}): {e}")
            continue
    
    print("\n" + "=" * 60)
    print("Image generation complete!")
    print(f"Generated: {generated_count}")
    print(f"Skipped (already exists): {skipped_count}")
    print(f"Total: {len(contexts)}")
    print(f"Images saved to: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()

