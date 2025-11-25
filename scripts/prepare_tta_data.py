import os
import sys
import json
import yaml
import hashlib
from tqdm import tqdm
from PIL import Image

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.tta_loader import TTALoader
from models.image_generator import ImageGenerator

def process_tta_dataset():
    # Load configuration
    config_path = "configs/config.yaml"
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    output_dir = os.path.join("outputs", "tta_images")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Loader and Generator
    try:
        loader = TTALoader(config_path)
        generator = ImageGenerator(config_path)
    except Exception as e:
        print(f"Initialization failed: {e}")
        return
    
    # Process text samples for multimodal expansion
    text_samples = loader.get_text_samples()
    print(f"Found {len(text_samples)} text samples to augment with images.")
    
    mapping = {}
    mapping_file = os.path.join("outputs", "tta_image_mapping.json")
    
    # Load existing mapping if available
    if os.path.exists(mapping_file):
        with open(mapping_file, "r", encoding="utf-8") as f:
            try:
                mapping = json.load(f)
            except json.JSONDecodeError:
                mapping = {}
            
    for i, sample in enumerate(tqdm(text_samples)):
        sample_id = sample.get('id')
        prompt = sample.get('input_prompt')
        
        if not prompt:
            continue
            
        # Create a safe filename using hash
        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        filename = f"{sample_id}_{prompt_hash}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        # Generate if not exists
        if not os.path.exists(filepath):
            try:
                # Use fixed seed for reproducibility
                # Using a hash of the prompt as a seed ensures consistent generation per prompt
                seed = int(hashlib.md5(prompt.encode('utf-8')).hexdigest(), 16) % (2**32)
                image = generator.generate(prompt, seed=seed)
                image.save(filepath)
            except Exception as e:
                print(f"Error generating image for {sample_id}: {e}")
                continue
        
        mapping[sample_id] = {
            "image_path": filepath,
            "prompt": prompt,
            "original_modality": "text"
        }
        
        # Save mapping periodically
        if i % 50 == 0:
            with open(mapping_file, "w", encoding="utf-8") as f:
                json.dump(mapping, f, ensure_ascii=False, indent=2)
                
    # Final save
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    
    print(f"Processing complete. Mapping saved to {mapping_file}")

if __name__ == "__main__":
    process_tta_dataset()
