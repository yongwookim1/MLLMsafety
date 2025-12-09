import os
import sys
import json
import yaml
import hashlib
import math
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.tta_loader import TTALoader
from models.image_generator import ImageGenerator

WORKER_FLUSH_INTERVAL = 32


def _append_mapping_records(worker_output_path, records):
    if not records:
        return
    os.makedirs(os.path.dirname(worker_output_path), exist_ok=True)
    with open(worker_output_path, 'a', encoding='utf-8') as f:
        for sample_id, payload in records.items():
            record = {"sample_id": sample_id}
            record.update(payload)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _collect_worker_results(worker_files):
    consolidated = {}
    for file_path in worker_files:
        if not os.path.exists(file_path):
            continue
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sample_id = record.pop('sample_id', None)
                if sample_id:
                    consolidated[sample_id] = record
    return consolidated


def _cleanup_worker_files(worker_files):
    for file_path in worker_files:
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass


def image_generation_worker(gpu_id, samples, config_path, output_dir, worker_output_path, flush_interval=WORKER_FLUSH_INTERVAL):
    """Worker process for generating images on a specific GPU"""
    local_mapping = {}

    def flush():
        if local_mapping:
            _append_mapping_records(worker_output_path, local_mapping)
            local_mapping.clear()

    try:
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else 'cpu'
        generator = None

        for sample in tqdm(samples, position=gpu_id, desc=f"GPU {gpu_id}"):
            sample_id = sample.get('id')
            prompt = sample.get('input_prompt')

            if not sample_id or not prompt:
                continue

            prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
            filename = f"{sample_id}_{prompt_hash}.jpg"
            filepath = os.path.join(output_dir, filename)

            if not os.path.exists(filepath):
                try:
                    if generator is None:
                        generator = ImageGenerator(config_path, device=device)
                    seed = int(prompt_hash, 16) % (2**32)
                    image = generator.generate(prompt, seed=seed)
                    image.save(filepath)
                except Exception as e:
                    print(f"Error generating image for {sample_id} on GPU {gpu_id}: {e}")
                    continue

            local_mapping[sample_id] = {
                'image_path': filepath,
                'prompt': prompt,
                'original_modality': 'text'
            }

            if len(local_mapping) >= flush_interval:
                flush()

    except Exception as e:
        print(f"Worker process on GPU {gpu_id} failed: {e}")
    finally:
        flush()


def _get_image_path(sample, output_dir):
    """Get the expected image path for a sample."""
    sample_id = sample.get('id')
    prompt = sample.get('input_prompt')
    if not sample_id or not prompt:
        return None
    prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
    filename = f"{sample_id}_{prompt_hash}.jpg"
    return os.path.join(output_dir, filename)


def _filter_samples_without_images(samples, output_dir):
    """Filter out samples that already have generated images."""
    samples_to_process = []
    skipped_count = 0
    
    for sample in samples:
        image_path = _get_image_path(sample, output_dir)
        if image_path is None:
            continue
        if os.path.exists(image_path):
            skipped_count += 1
        else:
            samples_to_process.append(sample)
    
    return samples_to_process, skipped_count


def _update_mapping_for_existing_images(samples, output_dir):
    """Update mapping file for samples with existing images."""
    mapping_file = os.path.join('outputs', 'tta_image_mapping.json')
    
    # Load existing mapping
    existing_mapping = {}
    if os.path.exists(mapping_file):
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                existing_mapping = json.load(f)
        except Exception:
            pass
    
    # Add entries for existing images
    updated = False
    for sample in samples:
        sample_id = sample.get('id')
        prompt = sample.get('input_prompt')
        image_path = _get_image_path(sample, output_dir)
        
        if image_path and os.path.exists(image_path) and sample_id not in existing_mapping:
            existing_mapping[sample_id] = {
                'image_path': image_path,
                'prompt': prompt,
                'original_modality': 'text'
            }
            updated = True
    
    if updated:
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(existing_mapping, f, ensure_ascii=False, indent=2)
        print(f"Updated mapping file: {mapping_file}")


def process_tta_dataset():
    config_path = 'configs/config.yaml'
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        _ = yaml.safe_load(f)

    output_dir = os.path.join('outputs', 'tta_images')
    os.makedirs(output_dir, exist_ok=True)

    try:
        loader = TTALoader(config_path)
    except Exception as e:
        print(f"Loader initialization failed: {e}")
        return

    text_samples = loader.get_text_samples()
    print(f"Found {len(text_samples)} text samples to augment with images.")
    
    # Filter out samples that already have images (before loading GPU)
    samples_to_process, skipped_count = _filter_samples_without_images(text_samples, output_dir)
    
    if skipped_count > 0:
        print(f"Skipping {skipped_count} samples with existing images.")
    
    if not samples_to_process:
        print("All images already exist. Nothing to generate.")
        # Still update mapping file with existing images
        _update_mapping_for_existing_images(text_samples, output_dir)
        return
    
    print(f"Processing {len(samples_to_process)} samples that need image generation.")

    mapping_file = os.path.join('outputs', 'tta_image_mapping.json')
    worker_files = []

    num_gpus = torch.cuda.device_count()

    if num_gpus > 1:
        print(f"🚀 Detected {num_gpus} GPUs. Starting parallel generation...")
        ctx = mp.get_context('spawn')
        processes = []
        chunk_size = max(1, math.ceil(len(samples_to_process) / num_gpus))

        for i in range(num_gpus):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(samples_to_process))
            chunk = samples_to_process[start_idx:end_idx]

            if not chunk:
                continue

            worker_file = os.path.join(output_dir, f'mapping_gpu_{i}.jsonl')
            if os.path.exists(worker_file):
                os.remove(worker_file)
            worker_files.append(worker_file)

            p = ctx.Process(
                target=image_generation_worker,
                args=(i, chunk, config_path, output_dir, worker_file)
            )
            p.start()
            processes.append(p)

        print('Waiting for workers to finish...')
        for p in processes:
            p.join()

    else:
        print('Running on single GPU/CPU...')
        worker_file = os.path.join(output_dir, 'mapping_gpu_0.jsonl')
        if os.path.exists(worker_file):
            os.remove(worker_file)
        worker_files.append(worker_file)
        image_generation_worker(0, samples_to_process, config_path, output_dir, worker_file)

    mapping = _collect_worker_results(worker_files)
    _cleanup_worker_files(worker_files)

    if os.path.exists(mapping_file):
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                old_mapping = json.load(f)
            old_mapping.update(mapping)
            mapping = old_mapping
        except Exception:
            pass

    # Also add skipped samples (existing images) to mapping
    for sample in text_samples:
        sample_id = sample.get('id')
        prompt = sample.get('input_prompt')
        image_path = _get_image_path(sample, output_dir)
        
        if sample_id and image_path and os.path.exists(image_path) and sample_id not in mapping:
            mapping[sample_id] = {
                'image_path': image_path,
                'prompt': prompt,
                'original_modality': 'text'
            }

    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"Processing complete. Mapping saved to {mapping_file}")


if __name__ == '__main__':
    process_tta_dataset()
