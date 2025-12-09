import os
import sys
import hashlib
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import yaml
import argparse

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from models import ImageGenerator
from data import KOBBQLoader


def main() -> None:
    """Generate images for KoBBQ dataset samples with various filtering options."""
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Generate images for KoBBQ dataset samples",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        help="Specific categories to generate images for (e.g., 'gender', 'race')"
    )
    parser.add_argument(
        "--bias-types",
        nargs="*",
        choices=["biased", "counter_biased"],
        help="Specific bias types to generate images for"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to generate"
    )
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Generate images for all unique samples instead of balanced sampling"
    )
    parser.add_argument(
        "--include-political",
        action="store_true",
        help="Include samples with political orientation content"
    )

    args = parser.parse_args()

    # Validate configuration file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)

    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("KoBBQ Image Generation Pipeline")
    print("=" * 40)

    # Initialize core components
    image_generator = ImageGenerator(args.config)
    data_loader = KOBBQLoader(args.config)

    # Extract relevant configuration sections
    gen_config = config["image_generation"]
    output_dir = gen_config["output_dir"]
    max_contexts = gen_config["max_contexts"]
    output_config = config["outputs"]
    mapping_file = output_config["image_context_mapping"]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load existing image mappings to avoid regeneration
    existing_contexts = load_existing_contexts(mapping_file)

    # Determine which samples to generate images for
    contexts = select_contexts_for_generation(
        data_loader, args, max_contexts
    )

    print(f"Total contexts to process: {len(contexts)}")
    print(f"Output directory: {output_dir}")
    print()

    # Run the image generation process
    run_image_generation(contexts, existing_contexts, image_generator, output_dir)


def load_existing_contexts(mapping_file: str) -> set:
    """
    Load set of contexts that already have generated images.

    Args:
        mapping_file: Path to JSON file containing context-to-image mappings

    Returns:
        Set of context strings that already exist
    """
    existing_contexts = set()

    if os.path.exists(mapping_file):
        print(f"Loading existing image-context mapping from {mapping_file}...")
        try:
            with open(mapping_file, "r", encoding="utf-8") as f:
                mapping = json.load(f)
                context_to_image = mapping.get("context_to_image", {})

                # Only count contexts where the image file actually exists
                for context, image_path in context_to_image.items():
                    if os.path.exists(image_path):
                        existing_contexts.add(context)

                print(f"Found {len(existing_contexts)} existing contexts with images")

        except Exception as e:
            print(f"Warning: Failed to load mapping file: {e}")

    return existing_contexts


def select_contexts_for_generation(data_loader, args, max_contexts: int):
    """
    Determine which contexts to generate images for based on command-line arguments.

    Args:
        data_loader: KOBBQLoader instance
        args: Parsed command-line arguments
        max_contexts: Default maximum number of contexts

    Returns:
        List of context dictionaries to process
    """
    # Determine sampling strategy and filters
    if args.all_samples:
        print("Loading all unique samples...")
        max_samples = args.max_samples or max_contexts
        contexts = data_loader.get_all_unique_samples(
            max_samples=max_samples,
            categories=args.categories,
            bias_types=args.bias_types
        )
    elif args.categories or args.bias_types:
        print(f"Loading samples for specific filters:")
        print(f"  Categories: {args.categories}")
        print(f"  Bias types: {args.bias_types}")
        max_samples = args.max_samples or max_contexts
        contexts = data_loader.get_all_unique_samples(
            max_samples=max_samples,
            categories=args.categories,
            bias_types=args.bias_types
        )
    else:
        print(f"Loading unique contexts (max: {max_contexts})...")
        contexts = data_loader.get_unique_contexts(max_contexts=max_contexts)

    # Apply political content filter if requested
    if args.include_political:
        contexts = filter_political_contexts(contexts)

    return contexts


def filter_political_contexts(contexts):
    """Filter contexts to include only those with political orientation content."""
    print("Filtering for political orientation related samples...")

    political_contexts = []
    for item in contexts:
        context_text = item["context"].lower()
        # Look for political keywords in Korean and English
        if "정치" in context_text or "political" in context_text:
            political_contexts.append(item)

    print(f"Found {len(political_contexts)} political orientation contexts")
    return political_contexts


def run_image_generation(contexts, existing_contexts: set, image_generator, output_dir: str) -> None:
    """Execute the main image generation loop."""
    generated_count = 0
    skipped_count = 0

    with tqdm(contexts, desc="Generating images") as progress_bar:
        for idx, item in enumerate(progress_bar):
            context = item["context"]
            sample_id = item["sample_id"]
            category = item.get("bbq_category", "Unknown")
            bias_type = item.get("bias_type", "Unknown")

            # Skip if context already has an image
            if context in existing_contexts:
                skipped_count += 1
                continue

            # Generate unique filename
            context_hash = hashlib.md5(context.encode()).hexdigest()[:8]
            safe_sample_id = sample_id.replace("/", "_").replace("\\", "_")
            filename = f"{safe_sample_id}_{context_hash}.jpg"
            save_image_path = os.path.join(output_dir, filename)

            # Ensure absolute path
            if not os.path.isabs(save_image_path):
                save_image_path = os.path.abspath(save_image_path)

            # Skip if file already exists (double-check)
            if os.path.exists(save_image_path):
                skipped_count += 1
                continue

            # Generate and save the image
            try:
                image = image_generator.generate(context)
                image.save(save_image_path, 'JPEG')
                generated_count += 1

                # Update progress display
                progress_bar.set_postfix({
                    'generated': generated_count,
                    'skipped': skipped_count,
                    'category': category,
                    'bias_type': bias_type
                })

            except Exception as e:
                print(f"\nError processing context (sample_id: {sample_id}, category: {category}): {e}")
                continue

    # Print final summary
    print("\nImage generation complete!")
    print(f"Generated: {generated_count}, Skipped: {skipped_count}, Total: {len(contexts)}")
    print(f"Images saved to: {output_dir}")

if __name__ == "__main__":
    main()

