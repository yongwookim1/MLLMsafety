import os
import sys
import json
import torch
import random
import hashlib
import numpy as np
import pandas as pd
import argparse
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Any
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPModel, CLIPProcessor

# Add parent directory to path to import modules from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.tta_loader import TTALoader

# Set random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class AlignmentEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Define Local Model Paths
        self.llm_path = os.path.abspath("./models_cache/qwen2.5-7b-instruct")
        self.image_model_path = os.path.abspath("./models_cache/qwen-image")
        
        # Note: KoCLIP might not be in models_cache, so we allow it to download if missing,
        # or you can map it to a local path if you have one.
        self.koclip_model_name = os.path.abspath("./models_cache/clip-vit-large-patch14-ko")
        
    def load_and_sample_data(self, target_count=500) -> List[Dict]:
        """Load TTA dataset and perform stratified sampling by category."""
        print("Loading TTA dataset...")
        loader = TTALoader()
        all_samples = loader.get_all_samples()
        
        if not all_samples:
            print("No samples loaded.")
            return []
            
        # Debug: Print first item keys to help identify correct fields
        # print(f"First item keys: {list(all_samples[0].keys())}")
        
        # Group by category
        category_groups = {}
        # Extended list of potential keys for text and category
        text_keys = ['input_prompt', 'prompt', 'text', 'question', 'instruction', 'input', 'content', 'user_prompt']
        cat_keys = ['risk', 'category', 'keyword', 'task', 'source', 'type', 'subcategory']

        for item in all_samples:
            text = None
            for k in text_keys:
                if item.get(k):
                    text = item.get(k)
                    break
            
            if not text: continue
                
            item['proc_text'] = text
            
            # Determine category
            cat = 'unknown'
            for k in cat_keys:
                if item.get(k):
                    cat = item.get(k)
                    break
            
            item['eval_category'] = cat
            
            if cat not in category_groups: category_groups[cat] = []
            category_groups[cat].append(item)
            
        print(f"Found {len(category_groups)} categories: {list(category_groups.keys())[:5]}...")
        
        # Stratified sampling
        sampled_data = []
        categories = list(category_groups.keys())
        if not categories: return []
            
        count_per_cat = target_count // len(categories)
        remainder = target_count % len(categories)
        
        for i, cat in enumerate(categories):
            sample_size = count_per_cat + (1 if i < remainder else 0)
            cat_items = category_groups[cat]
            selected = random.sample(cat_items, sample_size) if len(cat_items) >= sample_size else cat_items
            sampled_data.extend(selected)
            
        # Fill remainder
        if len(sampled_data) < target_count:
            remaining = [x for x in all_samples if x not in sampled_data and (x.get('prompt') or x.get('text'))]
            needed = target_count - len(sampled_data)
            if len(remaining) >= needed:
                sampled_data.extend(random.sample(remaining, needed))
            else:
                sampled_data.extend(remaining)
                
        print(f"Sampled {len(sampled_data)} items.")
        return sampled_data

    def translate_prompts(self, samples: List[Dict]) -> List[Dict]:
        """Translate Korean prompts to English using local Qwen2.5."""
        cache_file = os.path.join(self.args.output_dir, "translated_samples.json")
        
        # Try loading from cache
        if os.path.exists(cache_file):
            print(f"Loading cached translations from {cache_file}...")
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_samples = json.load(f)
                # Simple validation: check if count matches roughly or just use it
                # Since seed is fixed, we can assume samples are consistent if count matches
                if len(cached_samples) == len(samples):
                    print("Cache loaded successfully. Skipping translation.")
                    return cached_samples
                else:
                    print(f"Cache size mismatch ({len(cached_samples)} vs {len(samples)}). Re-running translation.")
            except Exception as e:
                print(f"Error loading cache: {e}")

        print(f"Loading Qwen2.5 LLM from {self.llm_path}...")
        
        if not os.path.exists(self.llm_path):
            print(f"Error: LLM not found at {self.llm_path}")
            return samples
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.llm_path, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.llm_path, 
                device_map="auto", 
                torch_dtype=torch.bfloat16,
                local_files_only=True
            )
        except Exception as e:
            print(f"Translation Model Load Error: {e}")
            return samples

        print("Translating prompts...")
        system_prompt = "You are a translator. Translate the Korean text to English concisely."
        
        batch_size = 64  # Increased batch size for 80GB GPU
        
        for i in tqdm(range(0, len(samples), batch_size), desc="Translating"):
            batch = samples[i:i+batch_size]
            batch_to_process = [item for item in batch if 'en_prompt' not in item]
            if not batch_to_process: continue
            
            texts = [item['proc_text'] for item in batch_to_process]
            
            prompts = []
            for t in texts:
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": t}]
                prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=256, 
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None
                )
                generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
                decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            for item, en_text in zip(batch_to_process, decoded):
                item['en_prompt'] = en_text.strip()
            
        del model, tokenizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # Save results to cache
        try:
            os.makedirs(self.args.output_dir, exist_ok=True)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)
            print(f"Translations saved to {cache_file}")
        except Exception as e:
            print(f"Error saving translation cache: {e}")

        return samples

    @staticmethod
    def _generate_worker(rank, chunk_data, output_dir, model_path, batch_size=4):
        """Worker for distributed generation with BATCH PROCESSING."""
        try:
            device_str = f"cuda:{rank}"
            print(f"[GPU {rank}] Loading model from {model_path}...")
            
            from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
            
            if not os.path.exists(model_path):
                 print(f"[GPU {rank}] Error: Model path not found: {model_path}")
                 return

            pipeline = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                local_files_only=True
            ).to(device_str)
            
            # pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
            pipeline.enable_attention_slicing(1)
            
            os.makedirs(output_dir, exist_ok=True)
            
            for i in tqdm(range(0, len(chunk_data), batch_size), desc=f"GPU {rank}", position=rank):
                batch_items = chunk_data[i:i+batch_size]
                needed_prompts = []
                needed_indices = []
                
                for idx, item in enumerate(batch_items):
                    if 'en_prompt' not in item: continue
                    prompt = item['en_prompt']
                    phash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
                    path = os.path.join(output_dir, f"en_{phash}.jpg")
                    if not os.path.exists(path):
                        needed_prompts.append(prompt)
                        needed_indices.append(idx)
                
                if not needed_prompts: continue
                    
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    images = pipeline(
                        prompt=needed_prompts, 
                        num_inference_steps=20,
                        guidance_scale=4.0
                    ).images
                
                for img, idx in zip(images, needed_indices):
                    prompt = batch_items[idx]['en_prompt']
                    phash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
                    path = os.path.join(output_dir, f"en_{phash}.jpg")
                    img.save(path)
            
            print(f"[GPU {rank}] Done.")
        except Exception as e:
            print(f"[GPU {rank}] Error: {e}")

    def generate_images_distributed(self, samples: List[Dict], batch_size=4):
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0: num_gpus = 1
        
        print(f"Generating images on {num_gpus} GPUs with Batch Size {batch_size}...")
        output_dir = os.path.join(self.args.output_dir, "comparison_en")
        
        valid_samples = [s for s in samples if 'en_prompt' in s]
        if not valid_samples:
            print("No valid English prompts to generate.")
            return

        # Check if any images need to be generated
        files_to_generate = []
        for s in valid_samples:
            phash = hashlib.md5(s['en_prompt'].encode('utf-8')).hexdigest()
            path = os.path.join(output_dir, f"en_{phash}.jpg")
            if not os.path.exists(path):
                files_to_generate.append(s)
        
        if not files_to_generate:
            print("All images already exist. Skipping generation.")
            return
            
        print(f"Need to generate {len(files_to_generate)} images out of {len(valid_samples)}.")
        
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        # Use only needed samples for distribution
        chunks = np.array_split(files_to_generate, num_gpus)
        chunks = [c.tolist() for c in chunks]
        
        # mp.set_start_method('spawn', force=True)  # Moved to main block to avoid runtime error
        procs = []
        for r in range(num_gpus):
            p = mp.Process(
                target=self._generate_worker, 
                args=(r, chunks[r], output_dir, self.image_model_path, batch_size)
            )
            p.start()
            procs.append(p)
        for p in procs: p.join()

    def evaluate_alignment(self, samples: List[Dict]):
        """Evaluate BOTH Korean and English images against the ORIGINAL KOREAN TEXT."""
        print(f"Starting evaluation with KoCLIP ({self.koclip_model_name})...")
        
        kr_img_dir = os.path.join(self.args.output_dir, "tta_images")
        en_img_dir = os.path.join(self.args.output_dir, "comparison_en")
        
        try:
            # Try loading local if possible, otherwise standard load
            # Note: KoCLIP usually isn't in our models_cache unless downloaded manually.
            # We assume internet access or pre-downloaded in HF cache.
            model = CLIPModel.from_pretrained(self.koclip_model_name, local_files_only=True).to(self.device)
            processor = CLIPProcessor.from_pretrained(self.koclip_model_name, local_files_only=True)
        except Exception as e:
            print(f"Failed to load KoCLIP: {e}")
            return

        results = []
        
        for item in tqdm(samples, desc="Evaluating"):
            kr_text = item['proc_text']
            category = item.get('eval_category', 'unknown')
            
            kr_hash = hashlib.md5(kr_text.encode('utf-8')).hexdigest()
            kr_path = os.path.join(kr_img_dir, f"{kr_hash}.jpg")
            
            en_text = item.get('en_prompt', "")
            if not en_text: continue
            en_hash = hashlib.md5(en_text.encode('utf-8')).hexdigest()
            en_path = os.path.join(en_img_dir, f"en_{en_hash}.jpg")
            
            score_kr_img = None
            score_en_img = None
            
            if os.path.exists(kr_path):
                try:
                    image = Image.open(kr_path).convert("RGB")
                    inputs = processor(text=[kr_text], images=image, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
                    with torch.no_grad():
                        score_kr_img = model(**inputs).logits_per_image.item()
                except Exception: pass
                
            if os.path.exists(en_path):
                try:
                    image = Image.open(en_path).convert("RGB")
                    inputs = processor(text=[kr_text], images=image, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
                    with torch.no_grad():
                        score_en_img = model(**inputs).logits_per_image.item()
                except Exception: pass
            
            if score_kr_img is not None and score_en_img is not None:
                results.append({
                    "category": category,
                    "kr_text": kr_text,
                    "score_kr_gen": score_kr_img,
                    "score_en_gen": score_en_img
                })
                
        if results:
            df = pd.DataFrame(results)
            
            print("\n" + "="*60)
            print(" Overall Alignment Results (Evaluated by KoCLIP against KR Text)")
            print("="*60)
            print(f" Total Samples: {len(df)}")
            print(f" Avg Score (KR Gen): {df['score_kr_gen'].mean():.4f}")
            print(f" Avg Score (EN Gen): {df['score_en_gen'].mean():.4f}")
            
            cat_summary = df.groupby("category")[["score_kr_gen", "score_en_gen"]].mean().reset_index()
            cat_summary["diff"] = cat_summary["score_en_gen"] - cat_summary["score_kr_gen"]
            
            print("\n" + "="*60)
            print(" Category-wise CLIP Scores")
            print("="*60)
            print(f"{'Category':<20} | {'KR Gen Score':<12} | {'EN Gen Score':<12}")
            print("-" * 60)
            for _, row in cat_summary.iterrows():
                print(f"{row['category'][:20]:<20} | {row['score_kr_gen']:.4f}       | {row['score_en_gen']:.4f}")
            print("="*60)
            
            out_csv = os.path.join(self.args.output_dir, "alignment_comparison_koclip.csv")
            df.to_csv(out_csv, index=False, encoding='utf-8-sig')
            
            out_summary = os.path.join(self.args.output_dir, "alignment_comparison_summary.csv")
            cat_summary.to_csv(out_summary, index=False, encoding='utf-8-sig')
            print(f"Detailed results saved to {out_csv}")
            print(f"Summary saved to {out_summary}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    
    evaluator = AlignmentEvaluator(args)
    
    # 1. Data Load
    samples = evaluator.load_and_sample_data(args.samples)
    if not samples: return
    
    # 2. Translate
    samples = evaluator.translate_prompts(samples)
    
    # 3. Generate
    evaluator.generate_images_distributed(samples, batch_size=args.batch_size)
    
    # 4. Evaluate
    evaluator.evaluate_alignment(samples)

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    main()