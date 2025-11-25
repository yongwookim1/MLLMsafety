import os
import sys
import hashlib
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import yaml
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from models import ImageGenerator
from data import KOBBQLoader

def main():
    parser = argparse.ArgumentParser(description="Generate images for KoBBQ dataset samples")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--categories", nargs="*", help="Specific categories to generate images for")
    parser.add_argument("--bias-types", nargs="*", help="Specific bias types to generate images for (biased, counter_biased)")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to generate")
    parser.add_argument("--all-samples", action="store_true", help="Generate images for all unique samples instead of balanced sampling")
    parser.add_argument("--include-political", action="store_true", help="Include samples with political content")

    args = parser.parse_args()

    config_path = args.config

    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("KOBBQ Image Generation")

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

    # Determine which samples to generate images for
    if args.all_samples:
        print("Loading all unique samples...")
        max_samples = args.max_samples or max_contexts
        contexts = data_loader.get_all_unique_samples(
            max_samples=max_samples,
            categories=args.categories,
            bias_types=args.bias_types
        )
    elif args.categories or args.bias_types:
        print(f"Loading samples for specific filters - categories: {args.categories}, bias_types: {args.bias_types}")
        max_samples = args.max_samples or max_contexts
        contexts = data_loader.get_all_unique_samples(
            max_samples=max_samples,
            categories=args.categories,
            bias_types=args.bias_types
        )
    else:
        print(f"Loading unique contexts (max: {max_contexts})...")
        contexts = data_loader.get_unique_contexts(max_contexts=max_contexts)

    # Filter for political content if requested
    if args.include_political:
        print("Filtering for political orientation related samples...")
        political_contexts = []
        for item in contexts:
            context = item["context"].lower()
            if "정치" in context or "political" in context:
                political_contexts.append(item)
        contexts = political_contexts
        print(f"Found {len(contexts)} political orientation related contexts")

    print(f"Total contexts to generate: {len(contexts)}")
    print(f"Output directory: {output_dir}")
    print()
    
    generated_count = 0
    skipped_count = 0
    
    for idx, item in enumerate(tqdm(contexts, desc="Generating images")):
        context = item["context"]
        sample_id = item["sample_id"]
        category = item.get("bbq_category", "Unknown")
        bias_type = item.get("bias_type", "Unknown")

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
                print(f"Progress: {idx + 1}/{len(contexts)} contexts processed")
                print(f"Generated: {generated_count}, Skipped: {skipped_count}")
                print(f"Current category: {category}, bias_type: {bias_type}")

        except Exception as e:
            print()
            print(f"Error processing context (sample_id: {sample_id}, category: {category}): {e}")
            continue
    
    print()
    print("Image generation complete")
    print(f"Generated: {generated_count}, Skipped: {skipped_count}, Total: {len(contexts)}")
    print(f"Images saved to: {output_dir}")

if __name__ == "__main__":
    main()

