#!/usr/bin/env python3

import os
import sys
import argparse
import json
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_requirements():
    """Check if required files exist based on config.yaml settings."""
    import yaml
    
    config_path = project_root / "configs" / "config.yaml"
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return False
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Get paths from config
    dataset_config = config.get("dataset", {})
    models_config = config.get("models", {})
    
    local_cache_dir = dataset_config.get("local_cache_dir", "./data_cache")
    image_gen_path = models_config.get("image_generator", {}).get("local_path", "")
    evaluator_path = models_config.get("evaluator", {}).get("local_path", "")
    
    required_files = [
        (os.path.join(local_cache_dir, "TTA01_AssurAI"), "TTA dataset"),
        (image_gen_path, "Image generator model"),
        (evaluator_path, "Evaluator model"),
    ]

    missing_files = []
    for file_path, desc in required_files:
        if file_path and not os.path.exists(file_path):
            missing_files.append(f"{file_path} ({desc})")

    if missing_files:
        print("Missing required files:")
        for file_info in missing_files:
            print(f"  - {file_info}")
        print("\nRun the following commands first:")
        print("1. python scripts/manual_download_tta.py")
        print("2. python scripts/download_models.py")
        return False

    print("All required files are present.")
    return True

def run_data_preparation():
    print("\n" + "="*50)
    print("Step 1: Converting text samples to images")
    print("="*50)

    from scripts.prepare_tta_data import process_tta_dataset

    try:
        process_tta_dataset()
        print("Data preparation completed")
        return True
    except Exception as e:
        print(f"Data preparation failed: {e}")
        return False

def run_evaluation(limit=None):
    print("\n" + "="*50)
    print("Step 2: Multimodal safety evaluation")
    print("="*50)

    from scripts.run_tta_evaluation import main as run_eval

    if limit:
        sys.argv = ['run_tta_evaluation.py', '--limit', str(limit)]
    else:
        sys.argv = ['run_tta_evaluation.py']

    try:
        run_eval()
        print("Evaluation completed")
        return True
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return False

def generate_summary():
    print("\n" + "="*50)
    print("Result Summary")
    print("="*50)

    evaluations_dir = "outputs/tta_results/evaluations"
    results_file = None
    judge_model = "unknown"

    if os.path.exists(evaluations_dir):
        judge_dirs = [d for d in os.listdir(evaluations_dir)
                     if os.path.isdir(os.path.join(evaluations_dir, d))]

        if judge_dirs:
            judge_dirs_with_time = [(d, os.path.getmtime(os.path.join(evaluations_dir, d)))
                                   for d in judge_dirs]
            judge_dirs_with_time.sort(key=lambda x: x[1], reverse=True)
            judge_model = judge_dirs_with_time[0][0]
            results_file = os.path.join(evaluations_dir, judge_model, "evaluation_results.json")
            print(f"Using results from judge model: {judge_model}")

    if not results_file or not os.path.exists(results_file):
        results_file = "outputs/tta_results/evaluation_results.json"
        print("Using legacy results path")

    if not os.path.exists(results_file):
        print("No evaluation results found.")
        return

    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    pipeline_summary_file = results_file.replace("evaluation_results.json", "pipeline_summary.json")
    os.makedirs(os.path.dirname(pipeline_summary_file), exist_ok=True)

    pipeline_summary = {
        "judge_model": judge_model,
        "total_evaluated_samples": len(results),
        "overall_stats": {},
        "category_averages": {}
    }

    text_scores = []
    multimodal_scores = []
    category_text_scores = defaultdict(list)
    category_multimodal_scores = defaultdict(list)

    for result in results:
        judge_results = result.get('judge_results', {})
        text_score = judge_results.get('text_only', {}).get('parsed_score')
        multimodal_score = judge_results.get('multimodal', {}).get('parsed_score')
        risk_category = result.get('risk_category', 'Unknown').split(':')[0]

        if text_score is not None:
            text_scores.append(text_score)
            category_text_scores[risk_category].append(text_score)
        if multimodal_score is not None:
            multimodal_scores.append(multimodal_score)
            category_multimodal_scores[risk_category].append(multimodal_score)

    pipeline_summary["overall_stats"] = {
        "text_only": {
            "count": len(text_scores),
            "avg_score": round(sum(text_scores) / len(text_scores), 2) if text_scores else None,
            "score_distribution": {f"score_{i}": text_scores.count(i) for i in range(0, 4)} if text_scores else {}
        },
        "multimodal": {
            "count": len(multimodal_scores),
            "avg_score": round(sum(multimodal_scores) / len(multimodal_scores), 2) if multimodal_scores else None,
            "score_distribution": {f"score_{i}": multimodal_scores.count(i) for i in range(0, 4)} if multimodal_scores else {}
        }
    }

    for category in set(list(category_text_scores.keys()) + list(category_multimodal_scores.keys())):
        text_scores_cat = category_text_scores.get(category, [])
        multimodal_scores_cat = category_multimodal_scores.get(category, [])

        pipeline_summary["category_averages"][category] = {
            "text_only": {
                "count": len(text_scores_cat),
                "avg_score": round(sum(text_scores_cat) / len(text_scores_cat), 2) if text_scores_cat else None
            },
            "multimodal": {
                "count": len(multimodal_scores_cat),
                "avg_score": round(sum(multimodal_scores_cat) / len(multimodal_scores_cat), 2) if multimodal_scores_cat else None
            }
        }

    with open(pipeline_summary_file, 'w', encoding='utf-8') as f:
        json.dump(pipeline_summary, f, ensure_ascii=False, indent=2)
    print(f"Pipeline summary saved to: {pipeline_summary_file}")

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")

    print(f"Judge Model: {pipeline_summary['judge_model']}")
    print(f"Total Samples: {pipeline_summary['total_evaluated_samples']}")

    overall = pipeline_summary.get('overall_stats', {})
    if overall.get('text_only'):
        text_stats = overall['text_only']
        print(f"\nText-Only Results:")
        print(f"  Samples: {text_stats['count']}")
        if text_stats.get('avg_score'):
            print(f"  Average Score: {text_stats['avg_score']:.2f}")
        if text_stats.get('score_distribution'):
            print("  Score Distribution:")
            for score_key, count in text_stats['score_distribution'].items():
                score = int(score_key.split('_')[1])
                pct = (count / text_stats['count']) * 100 if text_stats['count'] > 0 else 0
                print(f"    {score}: {count} ({pct:.1f}%)")

    if overall.get('multimodal'):
        multi_stats = overall['multimodal']
        print(f"\nMultimodal Results:")
        print(f"  Samples: {multi_stats['count']}")
        if multi_stats.get('avg_score'):
            print(f"  Average Score: {multi_stats['avg_score']:.2f}")
        if multi_stats.get('score_distribution'):
            print("  Score Distribution:")
            for score_key, count in multi_stats['score_distribution'].items():
                score = int(score_key.split('_')[1])
                pct = (count / multi_stats['count']) * 100 if multi_stats['count'] > 0 else 0
                print(f"    {score}: {count} ({pct:.1f}%)")

    categories = pipeline_summary.get('category_averages', {})
    if categories:
        print(f"\nTop 5 Risk Categories by Sample Count:")
        sorted_cats = sorted(categories.items(),
                           key=lambda x: x[1]['text_only']['count'] + x[1]['multimodal']['count'],
                           reverse=True)[:5]

        for cat_name, cat_data in sorted_cats:
            total_samples = cat_data['text_only']['count'] + cat_data['multimodal']['count']
            text_avg = cat_data['text_only'].get('avg_score', 'N/A')
            multi_avg = cat_data['multimodal'].get('avg_score', 'N/A')
            print(f"  {cat_name}: {total_samples} samples (Text: {text_avg}, Multi: {multi_avg})")

    print(f"\nDetailed results saved to: {pipeline_summary_file}")

    mapping_file = "outputs/tta_image_mapping.json"
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        print(f"Number of generated images: {len(mapping)}")

    print(f"\nResult file locations:")
    print(f"  - Evaluation results: {results_file}")
    summary_file = results_file.replace("evaluation_results.json", "evaluation_summary.json") if results_file else None
    if summary_file and os.path.exists(summary_file):
        print(f"  - Evaluation summary: {summary_file}")
    print(f"  - Image mapping: {mapping_file}")
    print(f"  - Generated images: outputs/tta_images/")

def main():
    parser = argparse.ArgumentParser(description="TTA Dataset Pipeline Execution")
    parser.add_argument("--skip-preparation", action="store_true",
                       help="Skip data preparation step")
    parser.add_argument("--skip-evaluation", action="store_true",
                       help="Skip evaluation step")
    parser.add_argument("--limit", type=int,
                       help="Maximum number of samples to evaluate (for testing)")
    parser.add_argument("--check-only", action="store_true",
                       help="Check requirements only, do not execute")

    args = parser.parse_args()

    print("TTA Dataset Multimodal Augmentation and Evaluation Pipeline")
    print("="*60)

    if not check_requirements():
        return 1

    if args.check_only:
        print("All checks completed. Pipeline is ready to run.")
        return 0

    success = True

    if not args.skip_preparation:
        if not run_data_preparation():
            success = False
    else:
        print("Skipping data preparation step")

    if not args.skip_evaluation and success:
        if not run_evaluation(args.limit):
            success = False
    else:
        print("Skipping evaluation step")

    if success:
        generate_summary()
        print("Pipeline execution completed successfully")
    else:
        print("Pipeline execution failed")

    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
