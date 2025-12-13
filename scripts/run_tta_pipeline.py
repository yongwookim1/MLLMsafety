#!/usr/bin/env python3

import os
import sys
import argparse
import json
import yaml
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, Any, List

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TTAPipelineRunner:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = project_root / config_path
        self.config = self._load_config()
        self.output_dir = Path("outputs/tta_results")
        self.evaluations_dir = self.output_dir / "evaluations"
        self.mapping_file = Path("outputs/tta_image_mapping.json")

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def check_requirements(self) -> bool:
        dataset_config = self.config.get("dataset", {})
        models_config = self.config.get("models", {})

        local_cache_dir = dataset_config.get("local_cache_dir", "./data_cache")
        image_gen_path = models_config.get("image_generator", {}).get("local_path", "")
        evaluator_path = models_config.get("evaluator", {}).get("local_path", "")

        required = [
            (os.path.join(local_cache_dir, "TTA01_AssurAI"), "TTA dataset"),
            (image_gen_path, "Image generator model"),
            (evaluator_path, "Evaluator model"),
        ]

        missing = [(p, d) for p, d in required if p and not os.path.exists(p)]
        if missing:
            print("Missing required files:")
            for path, desc in missing:
                print(f"  - {path} ({desc})")
            print("\nRun first:\n  1. python scripts/manual_download_tta.py\n  2. python scripts/download_models.py")
            return False

        print("All required files are present.")
        return True

    def run_data_preparation(self) -> bool:
        print("\n" + "=" * 50)
        print("Step 1: Converting text samples to images")
        print("=" * 50)

        try:
            from scripts.prepare_tta_data import process_tta_dataset
            process_tta_dataset()
            print("Data preparation completed")
            return True
        except Exception as e:
            print(f"Data preparation failed: {e}")
            return False

    def run_evaluation(self, limit: Optional[int] = None) -> bool:
        print("\n" + "=" * 50)
        print("Step 2: Multimodal safety evaluation")
        print("=" * 50)

        try:
            from scripts.run_tta_evaluation import main as run_eval
            sys.argv = ['run_tta_evaluation.py'] + (['--limit', str(limit)] if limit else [])
            run_eval()
            print("Evaluation completed")
            return True
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return False


class SummaryGenerator:
    def __init__(self, evaluations_dir: str = "outputs/tta_results/evaluations"):
        self.evaluations_dir = Path(evaluations_dir)

    def _find_latest_results(self) -> tuple[Optional[str], str]:
        if not self.evaluations_dir.exists():
            return None, "unknown"

        judge_dirs = [d for d in self.evaluations_dir.iterdir() if d.is_dir()]
        if not judge_dirs:
            return None, "unknown"

        latest = max(judge_dirs, key=lambda d: d.stat().st_mtime)
        results_file = latest / "evaluation_results.json"
        
        if results_file.exists():
            return str(results_file), latest.name
        return None, "unknown"

    def _load_results(self, results_file: str) -> List[Dict]:
        with open(results_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _compute_stats(self, results: List[Dict]) -> Dict[str, Any]:
        text_scores, multimodal_scores = [], []
        cat_text, cat_multi = defaultdict(list), defaultdict(list)

        for r in results:
            judge = r.get('judge_results', {})
            text_score = judge.get('text_only', {}).get('parsed_score')
            multi_score = judge.get('multimodal', {}).get('parsed_score')
            category = r.get('risk_category', 'Unknown').split(':')[0]

            if text_score is not None:
                text_scores.append(text_score)
                cat_text[category].append(text_score)
            if multi_score is not None:
                multimodal_scores.append(multi_score)
                cat_multi[category].append(multi_score)

        def score_stats(scores: List[int]) -> Dict[str, Any]:
            if not scores:
                return {"count": 0, "avg_score": None, "score_distribution": {}}
            return {
                "count": len(scores),
                "avg_score": round(sum(scores) / len(scores), 2),
                "score_distribution": {f"score_{i}": scores.count(i) for i in range(4)}
            }

        category_averages = {}
        for cat in set(cat_text.keys()) | set(cat_multi.keys()):
            category_averages[cat] = {
                "text_only": {"count": len(cat_text[cat]), "avg_score": round(sum(cat_text[cat]) / len(cat_text[cat]), 2) if cat_text[cat] else None},
                "multimodal": {"count": len(cat_multi[cat]), "avg_score": round(sum(cat_multi[cat]) / len(cat_multi[cat]), 2) if cat_multi[cat] else None}
            }

        return {
            "overall_stats": {"text_only": score_stats(text_scores), "multimodal": score_stats(multimodal_scores)},
            "category_averages": category_averages
        }

    def _print_summary(self, summary: Dict[str, Any]):
        print(f"\n{'=' * 60}")
        print("EVALUATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Judge Model: {summary['judge_model']}")
        print(f"Total Samples: {summary['total_evaluated_samples']}")

        for mode, label in [("text_only", "Text-Only"), ("multimodal", "Multimodal")]:
            stats = summary.get('overall_stats', {}).get(mode, {})
            if stats.get('count', 0) > 0:
                print(f"\n{label} Results:")
                print(f"  Samples: {stats['count']}")
                if stats.get('avg_score') is not None:
                    print(f"  Average Score: {stats['avg_score']:.2f}")
                if stats.get('score_distribution'):
                    print("  Score Distribution:")
                    for i in range(4):
                        cnt = stats['score_distribution'].get(f'score_{i}', 0)
                        pct = (cnt / stats['count']) * 100 if stats['count'] else 0
                        print(f"    {i}: {cnt} ({pct:.1f}%)")

        categories = summary.get('category_averages', {})
        if categories:
            print(f"\nTop 5 Risk Categories by Sample Count:")
            sorted_cats = sorted(categories.items(), key=lambda x: x[1]['text_only']['count'] + x[1]['multimodal']['count'], reverse=True)[:5]
            for cat, data in sorted_cats:
                total = data['text_only']['count'] + data['multimodal']['count']
                t_avg = data['text_only'].get('avg_score', 'N/A')
                m_avg = data['multimodal'].get('avg_score', 'N/A')
                print(f"  {cat}: {total} samples (Text: {t_avg}, Multi: {m_avg})")

    def generate(self):
        print("\n" + "=" * 50)
        print("Result Summary")
        print("=" * 50)

        results_file, judge_model = self._find_latest_results()
        
        # Fallback to legacy path
        if not results_file:
            legacy_path = "outputs/tta_results/evaluation_results.json"
            if os.path.exists(legacy_path):
                results_file = legacy_path
                print("Using legacy results path")
            else:
                print("No evaluation results found.")
                return

        print(f"Using results from judge model: {judge_model}")
        results = self._load_results(results_file)
        stats = self._compute_stats(results)

        summary = {
            "judge_model": judge_model,
            "total_evaluated_samples": len(results),
            **stats
        }

        # Save summary
        summary_file = Path(results_file).parent / "pipeline_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Pipeline summary saved to: {summary_file}")

        self._print_summary(summary)

        # Print file locations
        mapping_file = "outputs/tta_image_mapping.json"
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r', encoding='utf-8') as f:
                print(f"Number of generated images: {len(json.load(f))}")

        print(f"\nResult file locations:")
        print(f"  - Evaluation results: {results_file}")
        eval_summary = Path(results_file).parent / "evaluation_summary.json"
        if eval_summary.exists():
            print(f"  - Evaluation summary: {eval_summary}")
        print(f"  - Image mapping: {mapping_file}")
        print(f"  - Generated images: outputs/tta_images/")


def main():
    parser = argparse.ArgumentParser(description="TTA Dataset Pipeline Execution")
    parser.add_argument("--skip-preparation", action="store_true", help="Skip data preparation step")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation step")
    parser.add_argument("--limit", type=int, help="Maximum number of samples to evaluate")
    parser.add_argument("--check-only", action="store_true", help="Check requirements only")
    args = parser.parse_args()

    print("TTA Dataset Multimodal Augmentation and Evaluation Pipeline")
    print("=" * 60)

    runner = TTAPipelineRunner()

    if not runner.check_requirements():
        return 1

    if args.check_only:
        print("All checks completed. Pipeline is ready to run.")
        return 0

    success = True

    if not args.skip_preparation:
        success = runner.run_data_preparation()
    else:
        print("Skipping data preparation step")

    if success and not args.skip_evaluation:
        success = runner.run_evaluation(args.limit)
    elif args.skip_evaluation:
        print("Skipping evaluation step")

    if success:
        SummaryGenerator().generate()
        print("Pipeline execution completed successfully")
    else:
        print("Pipeline execution failed")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
