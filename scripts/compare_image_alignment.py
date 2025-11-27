import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import argparse
from PIL import Image
from tqdm import tqdm
from typing import List, Dict
from transformers import CLIPModel, CLIPProcessor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ImageAlignmentEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.koclip_model_path = os.path.abspath("./models_cache/clip-vit-large-patch14-ko")
        
        self.qwen_img_dir = os.path.join(args.output_dir, "qwen_images")
        self.kimchi_img_dir = os.path.join(args.output_dir, "kimchi_images")
        self.mapping_file = os.path.join(args.output_dir, "evaluation_results/qwen-image/image_context_mapping.json")
        
    def load_mapping(self) -> Dict:
        if not os.path.exists(self.mapping_file):
            print(f"Error: Mapping file not found: {self.mapping_file}")
            return {}
        
        with open(self.mapping_file, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        
        context_to_image = mapping.get("context_to_image", {})
        print(f"Loaded {len(context_to_image)} context-image mappings")
        return context_to_image

    def infer_category_from_filename(self, filename: str) -> str:
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

    def prepare_eval_pairs(self, context_to_image: Dict) -> List[Dict]:
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
            
            category = self.infer_category_from_filename(filename)
            
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

    def evaluate_alignment(self, eval_pairs: List[Dict]):
        print(f"\nLoading KoCLIP from {self.koclip_model_path}...")
        
        try:
            model = CLIPModel.from_pretrained(self.koclip_model_path, local_files_only=True).to(self.device)
            processor = CLIPProcessor.from_pretrained(self.koclip_model_path, local_files_only=True)
        except Exception as e:
            print(f"Failed to load KoCLIP: {e}")
            return
        
        results = []
        batch_size = self.args.batch_size
        
        print(f"\nEvaluating alignment (batch_size={batch_size})...")
        
        for i in tqdm(range(0, len(eval_pairs), batch_size), desc="Evaluating"):
            batch = eval_pairs[i:i+batch_size]
            
            texts = [b["context"] for b in batch]
            
            qwen_images, qwen_indices = [], []
            kimchi_images, kimchi_indices = [], []
            
            for idx, b in enumerate(batch):
                try:
                    if b["qwen_path"] and os.path.exists(b["qwen_path"]):
                        img = Image.open(b["qwen_path"]).convert("RGB")
                        qwen_images.append(img)
                        qwen_indices.append(idx)
                except Exception as e:
                    pass
                
                try:
                    if b["kimchi_path"] and os.path.exists(b["kimchi_path"]):
                        img = Image.open(b["kimchi_path"]).convert("RGB")
                        kimchi_images.append(img)
                        kimchi_indices.append(idx)
                except Exception as e:
                    pass
            
            qwen_scores = {}
            kimchi_scores = {}
            
            if qwen_images:
                try:
                    qwen_texts = [texts[idx] for idx in qwen_indices]
                    inputs = processor(
                        text=qwen_texts, images=qwen_images,
                        return_tensors="pt", padding=True, truncation=True, max_length=77
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        scores = outputs.logits_per_image.diag().cpu().numpy()
                    
                    for idx, score in zip(qwen_indices, scores):
                        qwen_scores[idx] = float(score)
                except Exception as e:
                    print(f"Qwen batch error: {e}")
            
            if kimchi_images:
                try:
                    kimchi_texts = [texts[idx] for idx in kimchi_indices]
                    inputs = processor(
                        text=kimchi_texts, images=kimchi_images,
                        return_tensors="pt", padding=True, truncation=True, max_length=77
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        scores = outputs.logits_per_image.diag().cpu().numpy()
                    
                    for idx, score in zip(kimchi_indices, scores):
                        kimchi_scores[idx] = float(score)
                except Exception as e:
                    print(f"Kimchi batch error: {e}")
            
            for idx, b in enumerate(batch):
                result = {
                    "category": b["category"],
                    "context": b["context"][:100],
                    "score_qwen": qwen_scores.get(idx),
                    "score_kimchi": kimchi_scores.get(idx)
                }
                results.append(result)
        
        self._save_results(results)

    def _save_results(self, results: List[Dict]):
        df = pd.DataFrame(results)
        
        df_qwen = df[df["score_qwen"].notna()].copy()
        df_kimchi = df[df["score_kimchi"].notna()].copy()
        df_both = df[(df["score_qwen"].notna()) & (df["score_kimchi"].notna())].copy()
        
        print("\n" + "="*70)
        print(" KoCLIP Alignment Results: Qwen-Image vs Kimchi")
        print("="*70)
        
        print(f"\n Total Samples: {len(df)}")
        print(f" Qwen images evaluated: {len(df_qwen)}")
        print(f" Kimchi images evaluated: {len(df_kimchi)}")
        print(f" Both available: {len(df_both)}")
        
        if len(df_qwen) > 0:
            print(f"\n [Qwen-Image] Avg Score: {df_qwen['score_qwen'].mean():.4f}")
        if len(df_kimchi) > 0:
            print(f" [Kimchi]     Avg Score: {df_kimchi['score_kimchi'].mean():.4f}")
        
        print("\n" + "="*70)
        print(" Category-wise CLIP Scores")
        print("="*70)
        print(f"{'Category':<25} | {'Qwen Score':<12} | {'Kimchi Score':<12} | {'Diff (K-Q)':<10}")
        print("-" * 70)
        
        category_results = []
        categories = sorted(df["category"].unique())
        
        for cat in categories:
            cat_df = df[df["category"] == cat]
            
            qwen_mean = cat_df["score_qwen"].mean() if cat_df["score_qwen"].notna().any() else None
            kimchi_mean = cat_df["score_kimchi"].mean() if cat_df["score_kimchi"].notna().any() else None
            
            diff = None
            if qwen_mean is not None and kimchi_mean is not None:
                diff = kimchi_mean - qwen_mean
            
            qwen_str = f"{qwen_mean:.4f}" if qwen_mean is not None else "N/A"
            kimchi_str = f"{kimchi_mean:.4f}" if kimchi_mean is not None else "N/A"
            diff_str = f"{diff:+.4f}" if diff is not None else "N/A"
            
            print(f"{cat[:25]:<25} | {qwen_str:<12} | {kimchi_str:<12} | {diff_str:<10}")
            
            category_results.append({
                "category": cat,
                "count": len(cat_df),
                "qwen_avg": round(qwen_mean, 4) if qwen_mean is not None else None,
                "kimchi_avg": round(kimchi_mean, 4) if kimchi_mean is not None else None,
                "diff_kimchi_minus_qwen": round(diff, 4) if diff is not None else None
            })
        
        print("="*70)
        
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        json_result = {
            "total_samples": len(df),
            "qwen_evaluated": len(df_qwen),
            "kimchi_evaluated": len(df_kimchi),
            "both_available": len(df_both),
            "overall": {
                "qwen_avg": round(df_qwen["score_qwen"].mean(), 4) if len(df_qwen) > 0 else None,
                "kimchi_avg": round(df_kimchi["score_kimchi"].mean(), 4) if len(df_kimchi) > 0 else None
            },
            "by_category": {r["category"]: r for r in category_results}
        }
        
        out_json = os.path.join(self.args.output_dir, "image_alignment_comparison.json")
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved: {out_json}")


def main():
    parser = argparse.ArgumentParser(description="Compare Qwen-Image vs Kimchi alignment using KoCLIP")
    parser.add_argument("--output_dir", default="./outputs", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    args = parser.parse_args()
    
    evaluator = ImageAlignmentEvaluator(args)
    
    context_to_image = evaluator.load_mapping()
    if not context_to_image:
        print("No mappings found. Exiting.")
        return
    
    eval_pairs = evaluator.prepare_eval_pairs(context_to_image)
    if not eval_pairs:
        print("No valid evaluation pairs. Exiting.")
        return
    
    evaluator.evaluate_alignment(eval_pairs)


if __name__ == "__main__":
    main()

