import os
import sys
import json
import torch
import random
import hashlib
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Any
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPModel, CLIPProcessor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.tta_loader import TTALoader

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class CLIPAlignmentEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.evaluation_mode = getattr(args, 'evaluation_mode', 'translation')

        # Model paths
        self.llm_path = os.path.abspath("./models_cache/qwen2.5-7b-instruct")
        self.image_model_path = os.path.abspath("./models_cache/qwen-image")
        self.koclip_model_path = os.path.abspath("./models_cache/clip-vit-large-patch14-ko")

        # Directory setup for translation mode
        self.kr_img_dir = os.path.join(self.args.output_dir, "tta_images")
        self.en_img_dir = os.path.join(self.args.output_dir, "comparison_en")
        self.mapping_file = os.path.join(self.args.output_dir, "tta_image_mapping.json")

    @staticmethod
    def _compute_sample_key(item: Dict[str, Any]) -> str:
        sample_id = item.get('id')
        if sample_id:
            return f"id::{sample_id}"
        proc_text = item.get('proc_text')
        if not proc_text:
            return None
        category = item.get('eval_category', '')
        source = f"{proc_text}|||{category}"
        return "hash::" + hashlib.sha1(source.encode('utf-8')).hexdigest()

    @staticmethod
    def _count_missing_translations(samples: List[Dict[str, Any]]) -> int:
        return sum(
            1
            for item in samples
            if item.get('proc_text') and not item.get('en_prompt')
        )

    def _apply_cached_translations(self, samples: List[Dict[str, Any]], translation_map: Dict[str, str]) -> int:
        if not translation_map:
            return 0
        applied = 0
        for item in samples:
            if not item.get('proc_text'):
                continue
            key = self._compute_sample_key(item)
            if not key:
                continue
            cached_prompt = translation_map.get(key)
            if cached_prompt:
                item['en_prompt'] = cached_prompt
                applied += 1
        return applied

    def _load_cached_translations(self, cache_file: str) -> Dict[str, str]:
        if not os.path.exists(cache_file):
            return {}
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to read translation cache: {e}")
            return {}

        translations = {}
        if isinstance(data, dict) and 'translations' in data:
            entries = data.get('translations', {})
            if isinstance(entries, dict):
                for key, value in entries.items():
                    if not key:
                        continue
                    if isinstance(value, dict):
                        text = value.get('en_prompt')
                    else:
                        text = value
                    if text:
                        translations[key] = text.strip()
        elif isinstance(data, list):
            print("Loaded legacy translation cache format. Re-indexing entries...")
            for item in data:
                if not isinstance(item, dict):
                    continue
                key = self._compute_sample_key(item)
                if not key:
                    continue
                text = item.get('en_prompt')
                if text:
                    translations[key] = text.strip()
        else:
            print("Warning: Unrecognized translation cache format. Ignoring cache.")
        return translations

    def load_and_sample_data(self, target_count=500) -> List[Dict]:
        return self._load_translation_data(target_count)

    def _load_translation_data(self, target_count=500) -> List[Dict]:
        print("Loading TTA dataset...")
        loader = TTALoader()
        all_samples = loader.get_all_samples()

        if not all_samples:
            print("No samples loaded.")
            return []

        category_groups = {}
        text_keys = ['input_prompt', 'prompt', 'text', 'question', 'instruction', 'input', 'content', 'user_prompt']
        cat_keys = ['risk', 'category', 'keyword', 'task', 'source', 'type', 'subcategory']

        processed_samples = []
        for item in all_samples:
            text = None
            for k in text_keys:
                if item.get(k):
                    text = item.get(k)
                    break

            if not text: continue

            item['proc_text'] = text

            cat = 'unknown'
            for k in cat_keys:
                if item.get(k):
                    cat = item.get(k)
                    break

            item['eval_category'] = cat
            processed_samples.append(item)

        print(f"Processed {len(processed_samples)} total samples")

        # Handle unlimited sampling (like Gemini script)
        if target_count == float('inf') or len(processed_samples) <= target_count:
            print(f"Using all {len(processed_samples)} samples")
            return processed_samples

        # Original sampling logic for limited count
        category_groups = {}
        for item in processed_samples:
            cat = item['eval_category']
            if cat not in category_groups:
                category_groups[cat] = []
            category_groups[cat].append(item)

        print(f"Found {len(category_groups)} categories: {list(category_groups.keys())[:5]}...")

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

        if len(sampled_data) < target_count:
            remaining = [x for x in processed_samples if x not in sampled_data and (x.get('prompt') or x.get('text'))]
            needed = target_count - len(sampled_data)
            if len(remaining) >= needed:
                sampled_data.extend(random.sample(remaining, needed))
            else:
                sampled_data.extend(remaining)

        print(f"Sampled {len(sampled_data)} items.")
        return sampled_data


    @staticmethod
    def _translation_worker(samples, llm_path, output_file):
        print(f"Loading Qwen2_5 LLM from {llm_path}...")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            if not os.path.exists(llm_path):
                print(f"Error: LLM not found at {llm_path}")
                return
            
            tokenizer = AutoTokenizer.from_pretrained(llm_path, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                llm_path, 
                device_map="auto", 
                torch_dtype=torch.bfloat16,
                local_files_only=True
            )
            
            print("Translating prompts...")
            system_prompt = "You are a translator. Translate the Korean text to English concisely."
            batch_size = 64
            
            items_to_process = []
            for idx, item in enumerate(samples):
                if 'en_prompt' not in item:
                    items_to_process.append((idx, item))
            
            if not items_to_process:
                print("Nothing to translate in worker.")
                return

            for i in tqdm(range(0, len(items_to_process), batch_size), desc="Translating"):
                batch_tuples = items_to_process[i:i+batch_size]
                batch_indices = [t[0] for t in batch_tuples]
                batch_items = [t[1] for t in batch_tuples]
                
                texts = [item['proc_text'] for item in batch_items]
                
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
                
                for idx, en_text in zip(batch_indices, decoded):
                    samples[idx]['en_prompt'] = en_text.strip()
                    
            translations = {}
            skipped = 0
            for item in samples:
                key = CLIPAlignmentEvaluator._compute_sample_key(item)
                if not key:
                    skipped += 1
                    continue
                en_prompt = item.get('en_prompt')
                if en_prompt:
                    translations[key] = en_prompt.strip()
            payload = {
                "schema_version": 2,
                "translations": translations,
                "total_entries": len(translations),
                "last_updated": datetime.utcnow().isoformat()
            }
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"Translations saved to {output_file}")
            if skipped:
                print(f"Note: {skipped} samples lacked cache keys and were not stored.")
            
        except Exception as e:
            print(f"Translation Worker Error: {e}")
            import traceback
            traceback.print_exc()

    def translate_prompts(self, samples: List[Dict]) -> List[Dict]:
        cache_file = os.path.join(self.args.output_dir, "translated_samples.json")
        translation_map = self._load_cached_translations(cache_file)
        if translation_map:
            applied = self._apply_cached_translations(samples, translation_map)
            print(f"Applied {applied} cached translations.")
        missing = self._count_missing_translations(samples)
        if missing == 0:
            print("All samples already contain English translations. Skipping translation.")
            return samples

        print(f"{missing} samples require translation. Starting translation in separate process...")
        
        p = mp.Process(
            target=self._translation_worker,
            args=(samples, self.llm_path, cache_file)
        )
        p.start()
        p.join()
        
        if p.exitcode != 0:
            print("Translation process failed.")
            return samples
        
        translation_map = self._load_cached_translations(cache_file)
        if translation_map:
            applied = self._apply_cached_translations(samples, translation_map)
            print(f"Applied {applied} translations from refreshed cache.")
        else:
            print("Warning: Translation cache missing after worker execution.")
        
        remaining = self._count_missing_translations(samples)
        if remaining:
            print(f"Warning: {remaining} samples still lack English translations.")
        return samples

    @staticmethod
    def _generate_worker(rank, chunk_data, output_dir, model_path, batch_size=1):
        try:
            import gc
            device_str = f"cuda:{rank}"
            print(f"[GPU {rank}] Loading model from {model_path}...")
            
            from diffusers import DiffusionPipeline
            
            if not os.path.exists(model_path):
                 print(f"[GPU {rank}] Error: Model path not found: {model_path}")
                 return

            pipeline = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                local_files_only=True
            ).to(device_str)
            
            pipeline.enable_attention_slicing(1)
            if hasattr(pipeline, 'enable_vae_slicing'):
                pipeline.enable_vae_slicing()
            
            os.makedirs(output_dir, exist_ok=True)
            
            for i in tqdm(range(0, len(chunk_data), batch_size), desc=f"GPU {rank}", position=rank):
                batch_items = chunk_data[i:i+batch_size]
                needed_prompts = []
                needed_indices = []
                
                for idx, item in enumerate(batch_items):
                    if 'en_prompt' not in item: continue
                    # Add Korean cultural context to the original prompt (like Gemini script)
                    prompt = item['en_prompt'] + ", Korean cultural context, authentic Korean aesthetic"
                    phash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
                    path = os.path.join(output_dir, f"en_{phash}.jpg")
                    if not os.path.exists(path):
                        needed_prompts.append(prompt)
                        needed_indices.append(idx)
                
                if not needed_prompts: continue
                    
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    images = pipeline(
                        prompt=needed_prompts, 
                        num_inference_steps=28,
                        guidance_scale=4.0
                    ).images
                
                for img, idx in zip(images, needed_indices):
                    prompt = batch_items[idx]['en_prompt']
                    phash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
                    path = os.path.join(output_dir, f"en_{phash}.jpg")
                    img.save(path)
                
                del images
                gc.collect()
                torch.cuda.empty_cache()
            
            print(f"[GPU {rank}] Done.")
        except Exception as e:
            print(f"[GPU {rank}] Error: {e}")
            import traceback
            traceback.print_exc()

    def generate_images_distributed(self, samples: List[Dict], batch_size=2):
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0: num_gpus = 1
        
        print(f"Generating images on {num_gpus} GPUs with Batch Size {batch_size}...")
        output_dir = os.path.join(self.args.output_dir, "comparison_en")
        
        valid_samples = [s for s in samples if 'en_prompt' in s]
        if not valid_samples:
            print("No valid English prompts to generate.")
            return

        files_to_generate = []
        existing_count = 0
        for s in valid_samples:
            phash = hashlib.md5(s['en_prompt'].encode('utf-8')).hexdigest()
            path = os.path.join(output_dir, f"en_{phash}.jpg")
            if not os.path.exists(path):
                files_to_generate.append(s)
            else:
                existing_count += 1

        print(f"Total valid samples: {len(valid_samples)}")
        print(f"Already generated: {existing_count}")
        print(f"Need to generate: {len(files_to_generate)}")

        if not files_to_generate:
            print("All images already exist. Skipping generation.")
            return
        
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        chunks = np.array_split(files_to_generate, num_gpus)
        chunks = [c.tolist() for c in chunks]
        
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
        if self.evaluation_mode == 'translation':
            return self._evaluate_translation_alignment(samples)
        else:
            raise ValueError(f"Unknown evaluation mode: {self.evaluation_mode}")

    def _get_completed_evaluation_ids(self, json_file: str) -> set:
        """Load IDs that have already been evaluated from the JSON file."""
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if 'results' in data:
                    return set(str(item.get('id', '')) for item in data['results'] if item.get('id'))
            except Exception as e:
                print(f"Warning: Could not read existing results: {e}")
        return set()

    def _print_evaluation_summary(self, json_file: str):
        """Read the full JSON and print evaluation summary."""
        if not os.path.exists(json_file):
            return

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            results = data.get('results', [])
            total = len(results)
            if total == 0:
                return

            print("\n" + "="*60)
            print(" Evaluation Summary (Including previous runs)")
            print("="*60)
            print(f" Total Evaluated: {total}")

            kr_scores = [r.get('score_kr_gen') for r in results if r.get('score_kr_gen') is not None]
            en_scores = [r.get('score_en_gen') for r in results if r.get('score_en_gen') is not None]

            if kr_scores and en_scores:
                print(f" Avg KR Score: {sum(kr_scores)/len(kr_scores):.4f}")
                print(f" Avg EN Score: {sum(en_scores)/len(en_scores):.4f}")
                diff = sum(en_scores)/len(en_scores) - sum(kr_scores)/len(kr_scores)
                print(f" Avg Difference (EN - KR): {diff:.4f}")

            print(f"Results saved to {json_file}")

        except Exception as e:
            print(f"Error generating summary: {e}")

    def _append_evaluation_results_to_json(self, new_results: List[Dict], json_file: str):
        """Append new evaluation results to the JSON file."""
        if not new_results: return

        # Load existing results
        existing_results = []
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    existing_results = data.get('results', [])
            except Exception as e:
                print(f"Warning: Could not read existing results: {e}")

        # Combine existing and new results
        all_results = existing_results + new_results

        # Save all results
        try:
            data = {
                "results": all_results,
                "total_samples": len(all_results),
                "last_updated": str(pd.Timestamp.now())
            }
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving results: {e}")

    def _evaluate_translation_alignment(self, samples: List[Dict]):
        print(f"Starting translation alignment evaluation with KoCLIP...")

        kr_img_dir = os.path.join(self.args.output_dir, "tta_images")
        en_img_dir = os.path.join(self.args.output_dir, "comparison_en")

        # Load image mapping for Korean images
        mapping_file = os.path.join(self.args.output_dir, "tta_image_mapping.json")
        kr_image_mapping = {}
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r', encoding='utf-8') as f:
                kr_image_mapping = json.load(f)
            print(f"Loaded {len(kr_image_mapping)} entries from image mapping")
        else:
            print(f"Warning: {mapping_file} not found")

        try:
            model = CLIPModel.from_pretrained(self.koclip_model_path, local_files_only=True).to(self.device)
            processor = CLIPProcessor.from_pretrained(self.koclip_model_path, local_files_only=True)
        except Exception as e:
            print(f"Failed to load KoCLIP: {e}")
            return

        # Check for resume capability
        out_json = os.path.join(self.args.output_dir, "alignment_comparison_koclip.json")
        completed_ids = self._get_completed_evaluation_ids(out_json)

        results = []
        batch_size = 2

        eval_items = []
        for item in samples:
            kr_text = item.get('proc_text')
            if not kr_text: continue

            en_prompt = item.get('en_prompt')
            if not en_prompt: continue

            sample_id = item.get('id')
            en_hash = hashlib.md5(en_prompt.encode('utf-8')).hexdigest()

            # Get Korean image path from mapping
            kr_path = None
            if sample_id and sample_id in kr_image_mapping:
                kr_path = kr_image_mapping[sample_id].get('image_path')

            en_path = os.path.join(en_img_dir, f"en_{en_hash}.jpg")

            if kr_path and os.path.exists(kr_path) and os.path.exists(en_path):
                eval_items.append({
                    'raw_item': item,
                    'kr_text': kr_text,
                    'kr_path': kr_path,
                    'en_path': en_path
                })

        # Filter out already completed evaluations
        to_evaluate = [item for item in eval_items if str(item['raw_item'].get('id', '')) not in completed_ids]

        print(f"Total potential pairs: {len(eval_items)}")
        print(f"Already completed: {len(completed_ids)}")
        print(f"Remaining to evaluate: {len(to_evaluate)}")
        print(f"Processing remaining pairs in batches of {batch_size}...")

        if not to_evaluate:
            print("All samples have already been evaluated!")
            self._print_evaluation_summary(out_json)
            return

        eval_items = to_evaluate  # Use filtered list

        batch_results = []  # For incremental saving
        save_interval = 5  # Save every 5 batches

        for i in tqdm(range(0, len(eval_items), batch_size), desc="Evaluating"):
            batch = eval_items[i:i+batch_size]

            texts = [b['kr_text'] for b in batch]

            kr_images = []
            en_images = []
            valid_indices = []

            for idx, b in enumerate(batch):
                try:
                    k_img = Image.open(b['kr_path']).convert("RGB")
                    e_img = Image.open(b['en_path']).convert("RGB")
                    kr_images.append(k_img)
                    en_images.append(e_img)
                    valid_indices.append(idx)
                except Exception as e:
                    print(f"Error loading image pair: {e}")
                    continue

            if not valid_indices: continue

            try:
                valid_texts = [texts[idx] for idx in valid_indices]

                inputs_kr = processor(
                    text=valid_texts,
                    images=kr_images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                ).to(self.device)

                with torch.no_grad():
                    outputs_kr = model(**inputs_kr)
                    logits_kr = outputs_kr.logits_per_image.diag().cpu().numpy()

                inputs_en = processor(
                    text=valid_texts,
                    images=en_images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                ).to(self.device)

                with torch.no_grad():
                    outputs_en = model(**inputs_en)
                    logits_en = outputs_en.logits_per_image.diag().cpu().numpy()

                batch_eval_results = []
                for v_idx, score_k, score_e in zip(valid_indices, logits_kr, logits_en):
                    original_item = batch[v_idx]['raw_item']
                    result = {
                        "id": original_item.get('id', f"unknown_{len(results)}"),
                        "category": original_item.get('eval_category', 'unknown'),
                        "kr_text": batch[v_idx]['kr_text'],
                        "score_kr_gen": float(score_k),
                        "score_en_gen": float(score_e)
                    }
                    results.append(result)
                    batch_eval_results.append(result)

                # Incremental save
                if batch_eval_results:
                    batch_results.extend(batch_eval_results)
                    if len(batch_results) >= save_interval:
                        self._append_evaluation_results_to_json(batch_results, out_json)
                        batch_results = []  # Reset batch

            except Exception as e:
                print(f"Batch evaluation error: {e}")
                continue

        # Save remaining batch results
        if batch_results:
            self._append_evaluation_results_to_json(batch_results, out_json)

        if results:
            df = pd.DataFrame(results)

        # Print final summary including all completed evaluations
        self._print_evaluation_summary(out_json)


def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP-based alignment using different modes")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples to evaluate. Set -1 for all available data.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--evaluation_mode", choices=['translation'],
                       default='translation', help="Evaluation mode to use")
    args = parser.parse_args()

    evaluator = CLIPAlignmentEvaluator(args)

    # Handle -1 for all samples (like Gemini script)
    target_count = float('inf') if args.samples == -1 else args.samples
    samples = evaluator.load_and_sample_data(target_count)
    if not samples: return

    print(f"Processing {len(samples)} samples")

    if evaluator.evaluation_mode == 'translation':
        samples = evaluator.translate_prompts(samples)
        evaluator.generate_images_distributed(samples, batch_size=args.batch_size)

    evaluator.evaluate_alignment(samples)

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
