import os
import sys
import json
import time
import random
import hashlib
import argparse
import torch
import numpy as np
import pandas as pd
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Any
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Allow importing from parent directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from data.tta_loader import TTALoader
except ImportError:
    TTALoader = None

# Set seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

class GeminiAlignmentEvaluator:
    def __init__(self, args):
        self.args = args
        self.setup_gemini()
        
        self.kr_img_dir = os.path.join(self.args.output_dir, "tta_images")
        self.en_img_dir = os.path.join(self.args.output_dir, "comparison_en")
        self.mapping_file = os.path.join(self.args.output_dir, "tta_image_mapping.json")
        self.image_model_path = os.path.abspath("./models_cache/qwen-image")
        
        # Output files
        self.out_csv = os.path.join(self.args.output_dir, "gemini_alignment_comparison.csv")
        self.out_json = os.path.join(self.args.output_dir, "gemini_alignment_comparison.json")

    def setup_gemini(self):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            api_key = input("Enter your Google Gemini API Key: ").strip()
        
        genai.configure(api_key=api_key)
        
        self.model_name = 'gemini-2.5-flash'
        print(f"Using Gemini Model: {self.model_name}")
        
        self.model = genai.GenerativeModel(self.model_name)
        
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

    @staticmethod
    def _generate_worker(rank, chunk_data, output_dir, model_path, batch_size=1):
        """Worker process to generate images using Qwen-Image (Flux based)"""
        try:
            import gc
            from diffusers import DiffusionPipeline
            
            device_str = f"cuda:{rank}"
            print(f"[GPU {rank}] Loading model from {model_path}...")
            
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
                    # Add Korean cultural context to the original prompt
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

    def generate_missing_images(self, samples: List[Dict]):
        """Checks for missing EN images and generates them using distributed GPUs."""
        
        missing_samples = []
        for s in samples:
            if 'en_prompt' not in s: continue
            phash = hashlib.md5(s['en_prompt'].encode('utf-8')).hexdigest()
            path = os.path.join(self.en_img_dir, f"en_{phash}.jpg")
            if not os.path.exists(path):
                missing_samples.append(s)
        
        if not missing_samples:
            print("All required English images exist.")
            return

        print(f"Generating {len(missing_samples)} missing images...")
        
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("Error: No GPUs available for image generation.")
            return

        chunks = np.array_split(missing_samples, num_gpus)
        chunks = [c.tolist() for c in chunks if len(c) > 0]
        
        active_gpus = len(chunks)
        procs = []
        for r in range(active_gpus):
            p = mp.Process(
                target=self._generate_worker, 
                args=(r, chunks[r], self.en_img_dir, self.image_model_path, self.args.batch_size)
            )
            p.start()
            procs.append(p)
        
        for p in procs: p.join()
        print("Image generation completed.")

    def load_dataset(self, target_count: int) -> List[Dict]:
        print("Loading dataset info...")
        all_samples = []
        
        # 1. Load from translated_samples.json
        trans_file = os.path.join(self.args.output_dir, "translated_samples.json")
        if os.path.exists(trans_file):
            try:
                with open(trans_file, 'r', encoding='utf-8') as f:
                    all_samples = json.load(f)
                print(f"Loaded {len(all_samples)} samples from translation file.")
            except Exception as e:
                print(f"Error reading translation file: {e}")

        # 2. Fallback to TTALoader
        if not all_samples and TTALoader:
            print("Falling back to TTALoader...")
            loader = TTALoader()
            all_samples = loader.get_all_samples()

        if not all_samples:
            print("Error: No samples found.")
            return []

        # Load mapping for KR images
        if not os.path.exists(self.mapping_file):
            print(f"Error: Mapping file not found at {self.mapping_file}")
            return []
            
        with open(self.mapping_file, 'r', encoding='utf-8') as f:
            kr_mapping = json.load(f)

        # Pre-scan existing English images
        existing_en_hashes = set()
        if os.path.exists(self.en_img_dir):
            for fname in os.listdir(self.en_img_dir):
                if fname.startswith("en_") and fname.endswith(".jpg"):
                    # fname format: en_{hash}.jpg
                    h = fname[3:-4]
                    existing_en_hashes.add(h)
        
        print(f"Found {len(existing_en_hashes)} existing English images in folder.")

        # Filter samples
        candidates = []
        for item in all_samples:
            sample_id = item.get('id')
            kr_text = item.get('proc_text')
            if not kr_text:
                for k in ['input_prompt', 'prompt', 'text', 'question', 'instruction', 'content']:
                    if item.get(k):
                        kr_text = item.get(k)
                        break
            
            if not kr_text or not sample_id: continue
            if sample_id not in kr_mapping: continue
            kr_path = kr_mapping[sample_id].get('image_path')
            if not kr_path or not os.path.exists(kr_path): continue

            en_prompt = item.get('en_prompt')
            if not en_prompt: continue
            
            en_hash = hashlib.md5(en_prompt.encode('utf-8')).hexdigest()
            
            # Check if English image exists
            # if en_hash not in existing_en_hashes:
            #     continue

            en_path = os.path.join(self.en_img_dir, f"en_{en_hash}.jpg")

            candidates.append({
                'id': sample_id,
                'text': kr_text,
                'en_prompt': en_prompt,
                'kr_path': kr_path,
                'en_path': en_path,
                'category': item.get('category', 'unknown')
            })

        print(f"Found {len(candidates)} candidate pairs (ready for evaluation or generation).")
        
        # Sampling
        if target_count > 0 and len(candidates) > target_count:
            # Determine priority: Prefer items that already have images OR items not yet evaluated?
            # Here we just stick to random sampling for consistency,
            # but the 'evaluate_pairs' method will check completed ones.
            selected_samples = random.sample(candidates, target_count)
            print(f"Randomly selected {len(selected_samples)} samples.")
            return selected_samples
        
        return candidates

    def _call_gemini_with_retry(self, prompt_parts, retries=3):
        """Call Gemini API with backoff. Returns None if failed."""
        for attempt in range(retries):
            try:
                response = self.model.generate_content(
                    prompt_parts, 
                    safety_settings=self.safety_settings
                )
                return response.text
            except Exception as e:
                print(f"API Error (Attempt {attempt+1}/{retries}): {e}")
                if attempt == retries - 1:
                    return None
                time.sleep(2 * (attempt + 1))
        return None

    def get_completed_ids(self):
        """Load IDs that have already been evaluated from the CSV file."""
        if os.path.exists(self.out_csv):
            try:
                df = pd.read_csv(self.out_csv)
                if 'id' in df.columns:
                    # Ensure IDs are strings for comparison
                    return set(df['id'].astype(str).tolist())
            except Exception as e:
                print(f"Warning: Could not read existing results: {e}")
        return set()

    def append_results_to_csv(self, new_results: List[Dict]):
        """Append new results to the CSV file immediately."""
        if not new_results: return
        
        df = pd.DataFrame(new_results)
        
        # If file doesn't exist, write with header. If exists, append without header.
        mode = 'a' if os.path.exists(self.out_csv) else 'w'
        header = not os.path.exists(self.out_csv)
        
        try:
            df.to_csv(self.out_csv, mode=mode, header=header, index=False, encoding='utf-8-sig')
        except Exception as e:
            print(f"Error saving incremental results: {e}")

    def evaluate_pairs(self, samples: List[Dict]):
        # 1. Generate missing images
        self.generate_missing_images(samples)
        
        # 2. Filter valid image pairs
        valid_samples = [s for s in samples if os.path.exists(s['en_path'])]
        if len(valid_samples) < len(samples):
            print(f"Warning: {len(samples) - len(valid_samples)} images failed to generate. Skipping.")

        # 3. Check Resume Capability
        completed_ids = self.get_completed_ids()
        to_evaluate = [s for s in valid_samples if str(s['id']) not in completed_ids]
        
        print(f"Total samples: {len(valid_samples)}")
        print(f"Already completed: {len(completed_ids)}")
        print(f"Remaining to evaluate: {len(to_evaluate)}")
        
        if not to_evaluate:
            print("All selected samples have already been evaluated!")
            self._print_summary_from_file()
            return

        # 4. Evaluation Loop
        print(f"Starting Gemini evaluation for {len(to_evaluate)} pairs...")
        
        batch_results = []
        save_interval = 10  # Save every 10 items
        
        pbar = tqdm(to_evaluate, desc="Evaluating")
        for i, item in enumerate(pbar):
            try:
                kr_img = Image.open(item['kr_path'])
                en_img = Image.open(item['en_path'])
                
                is_swapped = random.random() > 0.5
                img_a = en_img if is_swapped else kr_img
                img_b = kr_img if is_swapped else en_img
                
                prompt = [
                    # ------------------------------------------------------------------
                    # 1. System Role & Persona
                    # Defines a specific expert persona to align the evaluation context.
                    # ------------------------------------------------------------------
                    "System Role: You are an expert Visual Aesthetics Critic specializing in Korean culture, fashion, and social context.",
                    "Objective: Evaluate two images (A and B) to determine which one better reflects the 'Korean Style' and appearance based on the Original Text.",

                    # ------------------------------------------------------------------
                    # 2. Input Data
                    # ------------------------------------------------------------------
                    f"Original Text: \"{item['text']}\"",
                    "Image A:", img_a,
                    "Image B:", img_b,

                    # ------------------------------------------------------------------
                    # 3. Evaluation Rubric
                    # Explicitly defines what 'Korean Style' means, focusing on people and vibe.
                    # ------------------------------------------------------------------
                    "### Evaluation Rubric (Priority Order):",

                    "1. [Critical] Subject Appearance & Styling (The 'Korean Look'):",
                    "   - Do the people look authentically Korean? Focus on facial features, makeup style (e.g., 'K-Beauty' aesthetics), and hairstyles.",
                    "   - **Fashion:** Does the clothing match modern Korean trends (e.g., clean-cut, minimalist, street fashion) or historical accuracy (if applicable)?",
                    "   - **Differentiation:** Reject images that look like generic Asian stereotypes or clearly resemble Chinese/Japanese styling artifacts.",

                    "2. [High] Atmospheric Vibe & Environment:",
                    "   - Does the background capture the distinct 'vibe' of Korea (e.g., apartment complexes, neon alleyways, specific architectural tones)?",
                    "   - Note: Ignore illegible text strings, but evaluate the *visual placement* of signs to see if they match Korean street layouts.",

                    "3. [Medium] Contextual Relevance:",
                    "   - Does the image faithfully depict the situation described in the Original Text?",

                    # ------------------------------------------------------------------
                    # 4. Evaluation Process
                    # Forces a step-by-step analysis to prevent random guessing.
                    # ------------------------------------------------------------------
                    "### Evaluation Steps:",
                    "1. Analyze the facial features and styling of the main subjects in both images.",
                    "2. Check the background atmosphere for Korean cultural markers.",
                    "3. Compare which image feels more like a scene from a Korean drama, movie, or daily life.",
                    "4. Select the winner based on Cultural Authenticity.",

                    # ------------------------------------------------------------------
                    # 5. Output Format
                    # Returns strictly structured JSON for parsing.
                    # ------------------------------------------------------------------
                    "Respond ONLY in valid JSON format with no markdown blocks:",
                    "{",
                    "  \"analysis_people_A\": \"Assessment of facial features/styling in Image A.\",",
                    "  \"analysis_people_B\": \"Assessment of facial features/styling in Image B.\",",
                    "  \"comparison_reason\": \"Why one image looks more authentically Korean than the other (focus on people and vibe).\",",
                    "  \"choice\": \"A\", \"B\", or \"Tie\"",
                    "}"
                ]

                response_text = self._call_gemini_with_retry(prompt)
                if not response_text:
                    continue # Skip this item if API fails repeatedly

                clean_text = response_text.replace('```json', '').replace('```', '').strip()
                try:
                    eval_data = json.loads(clean_text)
                    choice = eval_data.get('choice', 'Tie').strip().upper()
                    reason = eval_data.get('comparison_reason', '')
                    analysis_a = eval_data.get('analysis_people_A', '')
                    analysis_b = eval_data.get('analysis_people_B', '')
                except json.JSONDecodeError:
                    print(f"JSON Error (ID {item['id']}): {clean_text[:50]}...")
                    continue

                winner = 'Tie'
                if choice == 'A':
                    winner = 'EN' if is_swapped else 'KR'
                elif choice == 'B':
                    winner = 'KR' if is_swapped else 'EN'

                res = {
                    "id": item['id'],
                    "category": item['category'],
                    "text": item['text'],
                    "winner": winner,
                    "raw_choice": choice,
                    "is_swapped": is_swapped,
                    "reason": reason,
                    "analysis_people_A": analysis_a,
                    "analysis_people_B": analysis_b
                }
                batch_results.append(res)

                # Incremental Save
                if len(batch_results) >= save_interval:
                    self.append_results_to_csv(batch_results)
                    batch_results = [] # Reset batch

            except Exception as e:
                print(f"Error processing item {item.get('id')}: {e}")
                continue
        
        # Save remaining results
        if batch_results:
            self.append_results_to_csv(batch_results)
            
        # Final Summary
        self._print_summary_from_file()

    def _print_summary_from_file(self):
        """Read the full CSV (including previous runs) and print stats."""
        if not os.path.exists(self.out_csv):
            return

        try:
            df = pd.read_csv(self.out_csv)
            total = len(df)
            kr_wins = len(df[df['winner'] == 'KR'])
            en_wins = len(df[df['winner'] == 'EN'])
            ties = len(df[df['winner'] == 'Tie'])
            
            print("\n" + "="*60)
            print(" Final Evaluation Summary (Including previous runs)")
            print("="*60)
            print(f" Total Evaluated: {total}")
            print(f" KR Wins (Original): {kr_wins} ({kr_wins/total*100:.1f}%)")
            print(f" EN Wins (Translated): {en_wins} ({en_wins/total*100:.1f}%)")
            print(f" Ties: {ties} ({ties/total*100:.1f}%)")
            print("-" * 60)
            
            # Save simplified JSON summary
            summary_data = {
                "summary": {"total": total, "kr_wins": kr_wins, "en_wins": en_wins, "ties": ties},
                "details": df.to_dict(orient='records')
            }
            with open(self.out_json, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Error generating summary: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--samples", type=int, default=500, help="Number of pairs to evaluate. Set -1 for all.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for image generation")
    args = parser.parse_args()
    
    # Fix for multiprocessing spawn
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
        
    evaluator = GeminiAlignmentEvaluator(args)
    
    # 1. Load data
    samples = evaluator.load_dataset(args.samples)
    
    if not samples:
        print("No valid sample pairs found to evaluate.")
        return
        
    # 2. Evaluate (Handles Resume & Incremental Save)
    evaluator.evaluate_pairs(samples)

if __name__ == "__main__":
    main()
