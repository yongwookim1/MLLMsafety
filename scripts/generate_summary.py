#!/usr/bin/env python3
"""Generate summary from existing evaluation results"""

import os
import sys
import json
from datetime import datetime
from collections import defaultdict

def generate_modality_comparison(results_file: str, output_dir: str):
    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    category_scores = defaultdict(lambda: {"text_only": [], "multimodal": []})
    
    for result in results:
        category = result.get('risk_category', 'Unknown').split(':')[0]
        judge_results = result.get('judge_results', {})
        
        text_score = judge_results.get('text_only', {}).get('parsed_score')
        multi_score = judge_results.get('multimodal', {}).get('parsed_score')
        
        if text_score is not None:
            category_scores[category]["text_only"].append(text_score)
        if multi_score is not None:
            category_scores[category]["multimodal"].append(multi_score)
    
    comparison = {
        "generated_at": datetime.now().isoformat(),
        "total_samples": len(results),
        "overall": {},
        "by_category": {}
    }
    
    all_text, all_multi = [], []
    
    for category in sorted(category_scores.keys()):
        scores = category_scores[category]
        text_scores = scores["text_only"]
        multi_scores = scores["multimodal"]
        
        all_text.extend(text_scores)
        all_multi.extend(multi_scores)
        
        cat_data = {
            "text_only": {
                "count": len(text_scores),
                "avg_score": round(sum(text_scores) / len(text_scores), 3) if text_scores else None,
                "score_distribution": {i: text_scores.count(i) for i in range(1, 6)} if text_scores else {}
            },
            "multimodal": {
                "count": len(multi_scores),
                "avg_score": round(sum(multi_scores) / len(multi_scores), 3) if multi_scores else None,
                "score_distribution": {i: multi_scores.count(i) for i in range(1, 6)} if multi_scores else {}
            }
        }
        
        if cat_data["text_only"]["avg_score"] and cat_data["multimodal"]["avg_score"]:
            cat_data["diff_multi_minus_text"] = round(
                cat_data["multimodal"]["avg_score"] - cat_data["text_only"]["avg_score"], 3
            )
        
        comparison["by_category"][category] = cat_data
    
    comparison["overall"] = {
        "text_only": {
            "count": len(all_text),
            "avg_score": round(sum(all_text) / len(all_text), 3) if all_text else None
        },
        "multimodal": {
            "count": len(all_multi),
            "avg_score": round(sum(all_multi) / len(all_multi), 3) if all_multi else None
        }
    }
    
    if comparison["overall"]["text_only"]["avg_score"] and comparison["overall"]["multimodal"]["avg_score"]:
        comparison["overall"]["diff_multi_minus_text"] = round(
            comparison["overall"]["multimodal"]["avg_score"] - comparison["overall"]["text_only"]["avg_score"], 3
        )
    
    output_file = os.path.join(output_dir, "modality_comparison.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    
    print(f"Saved: {output_file}")
    print(f"\n=== Overall Summary ===")
    print(f"Total samples: {len(results)}")
    print(f"Text-only avg: {comparison['overall']['text_only']['avg_score']}")
    print(f"Multimodal avg: {comparison['overall']['multimodal']['avg_score']}")
    if 'diff_multi_minus_text' in comparison['overall']:
        print(f"Difference (multi - text): {comparison['overall']['diff_multi_minus_text']}")
    
    print(f"\n=== By Category ===")
    for cat, data in comparison["by_category"].items():
        text_avg = data["text_only"]["avg_score"] or "N/A"
        multi_avg = data["multimodal"]["avg_score"] or "N/A"
        diff = data.get("diff_multi_minus_text", "N/A")
        print(f"{cat}: text={text_avg}, multi={multi_avg}, diff={diff}")

def main():
    results_file = "outputs/tta_results/evaluation_results.json"
    output_dir = "outputs/tta_results"
    
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found")
        sys.exit(1)
    
    generate_modality_comparison(results_file, output_dir)

if __name__ == "__main__":
    main()

