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

        # Directory setup based on mode
        if self.evaluation_mode == 'translation':
            self.kr_img_dir = os.path.join(self.args.output_dir, "tta_images")
            self.en_img_dir = os.path.join(self.args.output_dir, "comparison_en")
            self.mapping_file = os.path.join(self.args.output_dir, "tta_image_mapping.json")
        elif self.evaluation_mode == 'image_comparison':
            self.qwen_img_dir = os.path.join(args.output_dir, "qwen_images")
            self.kimchi_img_dir = os.path.join(args.output_dir, "kimchi_images")
            self.mapping_file = os.path.join(args.output_dir, "evaluation_results/qwen-image/image_context_mapping.json")

    def load_and_sample_data(self, target_count=500) -> List[Dict]:
        if self.evaluation_mode == 'translation':
            return self._load_translation_data(target_count)
        elif self.evaluation_mode == 'image_comparison':
            return self._load_image_comparison_data()
        else:
            raise ValueError(f"Unknown evaluation mode: {self.evaluation_mode}")

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

    def _load_image_comparison_data(self) -> List[Dict]:
        """Load data for image comparison evaluation (Qwen vs Kimchi)"""
        print("Loading image comparison data...")

        if not os.path.exists(self.mapping_file):
            print(f"Error: Mapping file not found: {self.mapping_file}")
            return []

        with open(self.mapping_file, 'r', encoding='utf-8') as f:
            mapping = json.load(f)

        context_to_image = mapping.get("context_to_image", {})
        print(f"Loaded {len(context_to_image)} context-image mappings")

        eval_pairs = []

        for context, qwen_path in context_to_image.items():
            if not context or not context.strip():
                continue

            filename = os.path.basename(qwen_path)
            kimchi_path = os.path.join(self.kimchi_img_dir, filename)
            qwen_full_path = os.path.join(self.qwen_img_dir, filename)

            qwen_exists = os.path.exists(qwen_full_path)
            kimchi_exists = os.path.exists(kimchi_path)

            if not qwen_exists and not kimchi_exists:
                continue

            category = self._infer_category_from_filename(filename)

            eval_pairs.append({
                "context": context,
                "qwen_path": qwen_full_path if qwen_exists else None,
                "kimchi_path": kimchi_path if kimchi_exists else None,
                "category": category,
                "filename": filename
            })

        print(f"Prepared {len(eval_pairs)} evaluation pairs")

        qwen_count = sum(1 for p in eval_pairs if p["qwen_path"])
        kimchi_count = sum(1 for p in eval_pairs if p["kimchi_path"])
        print(f"  - Qwen images: {qwen_count}")
        print(f"  - Kimchi images: {kimchi_count}")

        return eval_pairs

    def _infer_category_from_filename(self, filename: str) -> str:
        """Infer category from filename prefix"""
        prefix_map = {
            "age": "Age",
            "disability_status": "Disability_status",
            "gender_identity": "Gender_identity",
            "nationality": "Nationality",
            "physical_appearance": "Physical_appearance",
            "race_ethnicity": "Race_ethnicity",
            "religion": "Religion",
            "ses": "SES",
            "sexual_orientation": "Sexual_orientation",
        }

        fname_lower = filename.lower()
        for prefix, category in prefix_map.items():
            if fname_lower.startswith(prefix):
                return category
        return "Unknown"

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
            
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)
            print(f"Translations saved to {output_file}")
            
        except Exception as e:
            print(f"Translation Worker Error: {e}")
            import traceback
            traceback.print_exc()

    def translate_prompts(self, samples: List[Dict]) -> List[Dict]:
        cache_file = os.path.join(self.args.output_dir, "translated_samples.json")
        
        if os.path.exists(cache_file):
            print(f"Loading cached translations from {cache_file}...")
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_samples = json.load(f)
                if len(cached_samples) == len(samples):
                    print("Cache loaded successfully. Skipping translation.")
                    return cached_samples
                else:
                    print(f"Cache size mismatch ({len(cached_samples)} vs {len(samples)}). Re-running translation.")
            except Exception as e:
                print(f"Error loading cache: {e}")

        print("Starting translation in separate process...")
        
        p = mp.Process(
            target=self._translation_worker,
            args=(samples, self.llm_path, cache_file)
        )
        p.start()
        p.join()
        
        if p.exitcode != 0:
            print("Translation process failed.")
            return samples
        
        if os.path.exists(cache_file):
             with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
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
        elif self.evaluation_mode == 'image_comparison':
            return self._evaluate_image_comparison(samples)
        else:
            raise ValueError(f"Unknown evaluation mode: {self.evaluation_mode}")

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

        print(f"Found {len(eval_items)} pairs to evaluate. Processing in batches of {batch_size}...")

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

                for v_idx, score_k, score_e in zip(valid_indices, logits_kr, logits_en):
                    original_item = batch[v_idx]['raw_item']
                    results.append({
                        "category": original_item.get('eval_category', 'unknown'),
                        "kr_text": batch[v_idx]['kr_text'],
                        "score_kr_gen": float(score_k),
                        "score_en_gen": float(score_e)
                    })

            except Exception as e:
                print(f"Batch evaluation error: {e}")
                continue

        if results:
            df = pd.DataFrame(results)

            print("\n" + "="*60)
            print(" Translation Alignment Results (Evaluated by KoCLIP against KR Text)")
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

            # Save JSON results
            json_result = {
                "total_samples": len(df),
                "overall": {
                    "avg_score_kr_gen": round(df['score_kr_gen'].mean(), 4),
                    "avg_score_en_gen": round(df['score_en_gen'].mean(), 4),
                    "diff_en_minus_kr": round(df['score_en_gen'].mean() - df['score_kr_gen'].mean(), 4)
                },
                "by_category": {},
                "detailed_results": results
            }
            for _, row in cat_summary.iterrows():
                json_result["by_category"][row['category']] = {
                    "avg_score_kr_gen": round(row['score_kr_gen'], 4),
                    "avg_score_en_gen": round(row['score_en_gen'], 4),
                    "diff_en_minus_kr": round(row['diff'], 4)
                }

            out_json = os.path.join(self.args.output_dir, "alignment_comparison_koclip.json")
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(json_result, f, ensure_ascii=False, indent=2)

            print(f"Detailed results saved to {out_csv}")
            print(f"Summary saved to {out_summary}")
            print(f"JSON results saved to {out_json}")

    def _evaluate_image_comparison(self, eval_pairs: List[Dict]):
        """Evaluate alignment between Qwen and Kimchi generated images"""
        print(f"Starting image comparison evaluation with KoCLIP...")

        try:
            model = CLIPModel.from_pretrained(self.koclip_model_path, local_files_only=True).to(self.device)
            processor = CLIPProcessor.from_pretrained(self.koclip_model_path, local_files_only=True)
        except Exception as e:
            print(f"Failed to load KoCLIP: {e}")
            return

        results = []
        batch_size = 4

        print(f"Processing {len(eval_pairs)} pairs in batches of {batch_size}...")

        for i in tqdm(range(0, len(eval_pairs), batch_size), desc="Evaluating"):
            batch = eval_pairs[i:i+batch_size]

            qwen_images = []
            kimchi_images = []
            texts = []
            valid_items = []

            for item in batch:
                qwen_path = item.get('qwen_path')
                kimchi_path = item.get('kimchi_path')
                context = item.get('context', '')

                if not context:
                    continue

                qwen_img = None
                kimchi_img = None

                try:
                    if qwen_path and os.path.exists(qwen_path):
                        qwen_img = Image.open(qwen_path).convert("RGB")
                    if kimchi_path and os.path.exists(kimchi_path):
                        kimchi_img = Image.open(kimchi_path).convert("RGB")
                except Exception as e:
                    print(f"Error loading images: {e}")
                    continue

                # Need at least one image to evaluate
                if qwen_img is None and kimchi_img is None:
                    continue

                qwen_images.append(qwen_img)
                kimchi_images.append(kimchi_img)
                texts.append(context)
                valid_items.append(item)

            if not valid_items:
                continue

            try:
                # Evaluate Qwen images if available
                qwen_scores = []
                if any(img is not None for img in qwen_images):
                    valid_qwen_indices = [i for i, img in enumerate(qwen_images) if img is not None]
                    if valid_qwen_indices:
                        valid_qwen_texts = [texts[i] for i in valid_qwen_indices]
                        valid_qwen_images = [qwen_images[i] for i in valid_qwen_indices]

                        inputs_qwen = processor(
                            text=valid_qwen_texts,
                            images=valid_qwen_images,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=77
                        ).to(self.device)

                        with torch.no_grad():
                            outputs_qwen = model(**inputs_qwen)
                            qwen_logits = outputs_qwen.logits_per_image.diag().cpu().numpy()

                        # Map back to original indices
                        qwen_score_map = {}
                        for idx, score in zip(valid_qwen_indices, qwen_logits):
                            qwen_score_map[idx] = score

                        qwen_scores = [qwen_score_map.get(i, None) for i in range(len(texts))]

                # Evaluate Kimchi images if available
                kimchi_scores = []
                if any(img is not None for img in kimchi_images):
                    valid_kimchi_indices = [i for i, img in enumerate(kimchi_images) if img is not None]
                    if valid_kimchi_indices:
                        valid_kimchi_texts = [texts[i] for i in valid_kimchi_indices]
                        valid_kimchi_images = [kimchi_images[i] for i in valid_kimchi_indices]

                        inputs_kimchi = processor(
                            text=valid_kimchi_texts,
                            images=valid_kimchi_images,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=77
                        ).to(self.device)

                        with torch.no_grad():
                            outputs_kimchi = model(**inputs_kimchi)
                            kimchi_logits = outputs_kimchi.logits_per_image.diag().cpu().numpy()

                        # Map back to original indices
                        kimchi_score_map = {}
                        for idx, score in zip(valid_kimchi_indices, kimchi_logits):
                            kimchi_score_map[idx] = score

                        kimchi_scores = [kimchi_score_map.get(i, None) for i in range(len(texts))]

                # Store results
                for j, item in enumerate(valid_items):
                    result = {
                        "context": texts[j],
                        "category": item.get('category', 'Unknown'),
                        "filename": item.get('filename', ''),
                    }

                    if j < len(qwen_scores) and qwen_scores[j] is not None:
                        result["qwen_score"] = float(qwen_scores[j])
                    if j < len(kimchi_scores) and kimchi_scores[j] is not None:
                        result["kimchi_score"] = float(kimchi_scores[j])

                    results.append(result)

            except Exception as e:
                print(f"Batch evaluation error: {e}")
                continue

        if results:
            df = pd.DataFrame(results)

            print("\n" + "="*60)
            print(" Image Comparison Results (Qwen vs Kimchi)")
            print("="*60)
            print(f" Total Samples: {len(df)}")

            if 'qwen_score' in df.columns:
                print(f" Avg Qwen Score: {df['qwen_score'].mean():.4f}")
            if 'kimchi_score' in df.columns:
                print(f" Avg Kimchi Score: {df['kimchi_score'].mean():.4f}")

            if 'qwen_score' in df.columns and 'kimchi_score' in df.columns:
                both_scores = df.dropna(subset=['qwen_score', 'kimchi_score'])
                if len(both_scores) > 0:
                    print(f" Samples with both scores: {len(both_scores)}")
                    print(f" Qwen better: {(both_scores['qwen_score'] > both_scores['kimchi_score']).sum()}")
                    print(f" Kimchi better: {(both_scores['kimchi_score'] > both_scores['qwen_score']).sum()}")

            out_csv = os.path.join(self.args.output_dir, "image_comparison_results.csv")
            df.to_csv(out_csv, index=False, encoding='utf-8-sig')

            print(f"Results saved to {out_csv}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP-based alignment using different modes")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples to evaluate. Set -1 for all available data.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--evaluation_mode", choices=['translation', 'image_comparison'],
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
