import os
import sys
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.tta_evaluator import TTAEvaluationPipeline

def main():
    parser = argparse.ArgumentParser(description="Run TTA AssurAI Evaluation with Qwen-VL")
    parser.add_argument("--limit", type=int, help="Limit number of samples for testing", default=None)
    args = parser.parse_args()

    print("Initializing TTA Evaluation Pipeline...")
    try:
        pipeline = TTAEvaluationPipeline()
        print("Running evaluation...")
        pipeline.run_evaluation(limit=args.limit)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

