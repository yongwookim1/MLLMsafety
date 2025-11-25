import os
import sys
import json
import yaml
import hashlib
import math
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from PIL import Image

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.tta_loader import TTALoader
from models.image_generator import ImageGenerator

def image_generation_worker(gpu_id, samples, config_path, output_dir, queue):
    """Worker process for generating images on a specific GPU"""
    try:
        # Initialize generator on specific GPU
        generator = ImageGenerator(config_path, device=f"cuda:{gpu_id}")
        local_mapping = {}
        
        # Use position to prevent tqdm bars from overlapping
        for sample in tqdm(samples, position=gpu_id, desc=f"GPU {gpu_id}"):
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
                    seed = int(hashlib.md5(prompt.encode('utf-8')).hexdigest(), 16) % (2**32)
                    image = generator.generate(prompt, seed=seed)
                    image.save(filepath)
                except Exception as e:
                    print(f"Error generating image for {sample_id} on GPU {gpu_id}: {e}")
                    continue
            
            local_mapping[sample_id] = {
                "image_path": filepath,
                "prompt": prompt,
                "original_modality": "text"
            }
            
        queue.put(local_mapping)
        
    except Exception as e:
        print(f"Worker process on GPU {gpu_id} failed: {e}")
        queue.put({})

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
    
    # Initialize Loader
    try:
        loader = TTALoader(config_path)
    except Exception as e:
        print(f"Loader initialization failed: {e}")
        return
    
    # Process text samples
    text_samples = loader.get_text_samples()
    print(f"Found {len(text_samples)} text samples to augment with images.")
    
    mapping = {}
    mapping_file = os.path.join("outputs", "tta_image_mapping.json")
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    
    if num_gpus > 1:
        print(f"🚀 Detected {num_gpus} GPUs. Starting parallel generation...")
        
        # Use spawn start method for PyTorch CUDA compatibility
        ctx = mp.get_context('spawn')
        queue = ctx.Queue()
        processes = []
        
        # Split samples into chunks
        chunk_size = math.ceil(len(text_samples) / num_gpus)
        
        for i in range(num_gpus):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(text_samples))
            chunk = text_samples[start_idx:end_idx]
            
            if not chunk:
                continue
                
            p = ctx.Process(
                target=image_generation_worker,
                args=(i, chunk, config_path, output_dir, queue)
            )
            p.start()
            processes.append(p)
        
        # Collect results
        print("Waiting for workers to finish...")
        for _ in range(len(processes)):
            local_map = queue.get()
            mapping.update(local_map)
            
        for p in processes:
            p.join()
            
    else:
        print("Running on single GPU/CPU...")
        # Fallback to single process logic (using the worker function for simplicity)
        queue = mp.Queue()
        image_generation_worker(0, text_samples, config_path, output_dir, queue)
        mapping = queue.get()

    # Load existing mapping to merge if needed (optional, based on requirements)
    # Here we simply overwrite/save the current run's result or merge with old file
    if os.path.exists(mapping_file):
        try:
            with open(mapping_file, "r", encoding="utf-8") as f:
                old_mapping = json.load(f)
                # Update old mapping with new results
                old_mapping.update(mapping)
                mapping = old_mapping
        except:
            pass

    # Final save
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    
    print(f"Processing complete. Mapping saved to {mapping_file}")

if __name__ == "__main__":
    process_tta_dataset()
