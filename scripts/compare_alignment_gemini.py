import os
import sys
import json
import time
import random
import hashlib
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Any
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Allow importing from parent directories (matching original style)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from data.tta_loader import TTALoader
except ImportError:
    # Fallback if running standalone without project structure
    TTALoader = None

# Set seeds for reproducibility
random.seed(42)

class GeminiAlignmentEvaluator:
    def __init__(self, args):
        self.args = args
        self.setup_gemini()
        
        self.kr_img_dir = os.path.join(self.args.output_dir, "tta_images")
        self.en_img_dir = os.path.join(self.args.output_dir, "comparison_en")
        self.mapping_file = os.path.join(self.args.output_dir, "tta_image_mapping.json")

    def setup_gemini(self):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            # Allow user to input key if env var is missing
            api_key = input("Enter your Google Gemini API Key: ").strip()
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash') # or gemini-1.5-pro
        
        # Safety settings to avoid blocking evaluation of sensitive content
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

    def load_dataset(self, target_count=100) -> List[Dict]:
        """
        Loads data and matches it with existing images.
        Requires TTALoader or a compatible list of samples.
        """
        print("Loading TTA dataset info...")
        if TTALoader:
            loader = TTALoader()
            all_samples = loader.get_all_samples()
        else:
            print("Warning: TTALoader not found. Please ensure data is accessible.")
            return []

        # Load mapping to find KR images
        if not os.path.exists(self.mapping_file):
            print(f"Error: Mapping file not found at {self.mapping_file}")
            return []
            
        with open(self.mapping_file, 'r', encoding='utf-8') as f:
            kr_mapping = json.load(f)

        # Load cached translations (to get EN prompts)
        trans_file = os.path.join(self.args.output_dir, "translated_samples.json")
        if os.path.exists(trans_file):
            with open(trans_file, 'r', encoding='utf-8') as f:
                # Update all_samples with translation info if available
                trans_map = {item['id']: item for item in json.load(f) if 'id' in item}
                for item in all_samples:
                    if item.get('id') in trans_map:
                        item['en_prompt'] = trans_map[item['id']].get('en_prompt')

        # Filter paired samples
        valid_pairs = []
        print("Checking for valid image pairs (KR & EN)...")
        
        for item in all_samples:
            sample_id = item.get('id')
            kr_text = None
            
            # Extract text similar to original script
            text_keys = ['input_prompt', 'prompt', 'text', 'question', 'instruction', 'content']
            for k in text_keys:
                if item.get(k):
                    kr_text = item.get(k)
                    break
            
            if not kr_text or not sample_id: continue

            # Check KR Image existence
            if sample_id not in kr_mapping: continue
            kr_path = kr_mapping[sample_id].get('image_path')
            if not kr_path or not os.path.exists(kr_path): continue

            # Check EN Image existence
            en_prompt = item.get('en_prompt')
            if not en_prompt: continue
            en_hash = hashlib.md5(en_prompt.encode('utf-8')).hexdigest()
            en_path = os.path.join(self.en_img_dir, f"en_{en_hash}.jpg")
            if not os.path.exists(en_path): continue

            valid_pairs.append({
                'id': sample_id,
                'text': kr_text,
                'en_prompt': en_prompt,
                'kr_path': kr_path,
                'en_path': en_path,
                'category': item.get('category', 'unknown')
            })

        print(f"Found {len(valid_pairs)} valid pairs.")
        
        # Random sampling if needed
        if len(valid_pairs) > target_count:
            return random.sample(valid_pairs, target_count)
        return valid_pairs

    def _call_gemini_with_retry(self, prompt_parts, retries=3):
        for attempt in range(retries):
            try:
                response = self.model.generate_content(
                    prompt_parts, 
                    safety_settings=self.safety_settings
                )
                return response.text
            except Exception as e:
                if attempt == retries - 1:
                    print(f"Gemini API Failed after {retries} attempts: {e}")
                    return None
                time.sleep(2 * (attempt + 1))
        return None

    def evaluate_pairs(self, samples: List[Dict]):
        print(f"Starting Gemini evaluation for {len(samples)} pairs...")
        results = []
        
        pbar = tqdm(samples, desc="Evaluating")
        for item in pbar:
            try:
                kr_img = Image.open(item['kr_path'])
                en_img = Image.open(item['en_path'])
                
                # FAIRNESS: Randomize order to prevent position bias
                # is_swapped=True means Image A is EN, Image B is KR
                is_swapped = random.random() > 0.5
                
                img_a = en_img if is_swapped else kr_img
                img_b = kr_img if is_swapped else en_img
                
                prompt = [
                    "You are an expert image quality evaluator focusing on semantic alignment.",
                    f"Original Text: \"{item['text']}\"",
                    "Compare Image A and Image B. Which image better reflects the meaning, details, and nuance of the Original Text?",
                    "Ignore aesthetic quality (style, resolution) and focus ONLY on how well the content matches the text.",
                    "Image A:", img_a,
                    "Image B:", img_b,
                    "Respond in JSON format with two keys:",
                    "- 'choice': 'A', 'B', or 'Tie'",
                    "- 'reason': 'Short explanation of why.'",
                    "Do not use markdown code blocks."
                ]

                response_text = self._call_gemini_with_retry(prompt)
                
                if not response_text:
                    continue

                # Parse JSON roughly
                clean_text = response_text.replace('```json', '').replace('```', '').strip()
                try:
                    eval_data = json.loads(clean_text)
                    choice = eval_data.get('choice', 'Tie').strip().upper()
                    reason = eval_data.get('reason', '')
                except json.JSONDecodeError:
                    print(f"JSON Parse Error: {clean_text}")
                    continue

                # Map back to KR vs EN
                # If not swapped: A=KR, B=EN. If choice A -> KR Win.
                # If swapped: A=EN, B=KR. If choice A -> EN Win.
                winner = 'Tie'
                if choice == 'A':
                    winner = 'EN' if is_swapped else 'KR'
                elif choice == 'B':
                    winner = 'KR' if is_swapped else 'EN'

                results.append({
                    "id": item['id'],
                    "category": item['category'],
                    "text": item['text'],
                    "winner": winner,
                    "raw_choice": choice,
                    "is_swapped": is_swapped,
                    "reason": reason
                })
                
            except Exception as e:
                print(f"Error processing item {item.get('id')}: {e}")
                continue

        self._save_results(results)

    def _save_results(self, results: List[Dict]):
        if not results:
            print("No results to save.")
            return

        df = pd.DataFrame(results)
        
        # Calculate stats
        total = len(df)
        kr_wins = len(df[df['winner'] == 'KR'])
        en_wins = len(df[df['winner'] == 'EN'])
        ties = len(df[df['winner'] == 'Tie'])
        
        print("\n" + "="*60)
        print(" Gemini Evaluation Results")
        print("="*60)
        print(f" Total Evaluated: {total}")
        print(f" KR Wins (Original): {kr_wins} ({kr_wins/total*100:.1f}%)")
        print(f" EN Wins (Translated): {en_wins} ({en_wins/total*100:.1f}%)")
        print(f" Ties: {ties} ({ties/total*100:.1f}%)")
        print("-" * 60)
        
        # Category breakdown
        if 'category' in df.columns:
            cat_grp = df.groupby(['category', 'winner']).size().unstack(fill_value=0)
            print("\nBreakdown by Category:")
            print(cat_grp)
            cat_csv = os.path.join(self.args.output_dir, "gemini_category_summary.csv")
            cat_grp.to_csv(cat_csv)

        # Save details
        out_csv = os.path.join(self.args.output_dir, "gemini_alignment_comparison.csv")
        df.to_csv(out_csv, index=False, encoding='utf-8-sig')
        
        out_json = os.path.join(self.args.output_dir, "gemini_alignment_comparison.json")
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": {"total": total, "kr_wins": kr_wins, "en_wins": en_wins, "ties": ties},
                "details": results
            }, f, ensure_ascii=False, indent=2)
            
        print(f"\nDetailed results saved to {out_csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--samples", type=int, default=100, help="Number of pairs to evaluate via API")
    args = parser.parse_args()
    
    evaluator = GeminiAlignmentEvaluator(args)
    
    # 1. Load data and verify pairs exist
    samples = evaluator.load_dataset(args.samples)
    
    if not samples:
        print("No valid sample pairs found to evaluate.")
        return
        
    # 2. Run comparison
    evaluator.evaluate_pairs(samples)

if __name__ == "__main__":
    main()

