#!/usr/bin/env python3

import os
import sys
import argparse
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_requirements():
    required_files = [
        "data_cache/TTA01_AssurAI/data-00000-of-00001.arrow",
        "models_cache/qwen-image",
        "models_cache/qwen2.5-vl-7b-instruct"
    ]

    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print("Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
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

    results_file = "outputs/tta_results/evaluation_results.json"
    mapping_file = "outputs/tta_image_mapping.json"

    if os.path.exists(results_file):
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

        print(f"Number of evaluated samples: {len(results)}")

        text_only_scores = []
        multimodal_scores = []
        for result in results:
            judge_results = result.get('judge_results', {})
            text_score = judge_results.get('text_only', {}).get('parsed_score')
            multimodal_score = judge_results.get('multimodal', {}).get('parsed_score')

            if text_score is not None:
                text_only_scores.append(text_score)
            if multimodal_score is not None:
                multimodal_scores.append(multimodal_score)

        if text_only_scores:
            avg_text_score = sum(text_only_scores) / len(text_only_scores)
            print(f"Average text-only score: {avg_text_score:.2f}")

            text_counts = {}
            for score in text_only_scores:
                text_counts[score] = text_counts.get(score, 0) + 1

            print("Text-only score distribution:")
            for score in sorted(text_counts.keys()):
                count = text_counts[score]
                percentage = (count / len(text_only_scores)) * 100
                print(f"  Score {score}: {count} samples ({percentage:.1f}%)")

        if multimodal_scores:
            avg_multimodal_score = sum(multimodal_scores) / len(multimodal_scores)
            print(f"Average multimodal score: {avg_multimodal_score:.2f}")

            multimodal_counts = {}
            for score in multimodal_scores:
                multimodal_counts[score] = multimodal_counts.get(score, 0) + 1

            print("Multimodal score distribution:")
            for score in sorted(multimodal_counts.keys()):
                count = multimodal_counts[score]
                percentage = (count / len(multimodal_scores)) * 100
                print(f"  Score {score}: {count} samples ({percentage:.1f}%)")

    if os.path.exists(mapping_file):
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping = json.load(f)

        print(f"Number of generated images: {len(mapping)}")

    print(f"\nResult file locations:")
    print(f"  - Evaluation results: {results_file}")
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
