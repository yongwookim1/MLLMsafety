import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from models import ImageGenerator
from data import KOBBQLoader
from evaluation import EvaluationPipeline

def main():
    config_path = "configs/config.yaml"
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("Full MLLM Safety Evaluation Pipeline")
    print("=" * 60)
    
    steps = [
        ("1. Checking models", check_models),
        ("2. Generating images", generate_images),
        ("3. Creating image-context mapping", create_mapping),
        ("4. Running comparison", run_comparison),
        ("5. Evaluating results", evaluate_results),
        ("6. Finding mismatch cases", find_mismatches)
    ]
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        print("-" * 60)
        try:
            step_func(config_path)
            print(f"✓ {step_name} completed")
        except Exception as e:
            print(f"✗ Error in {step_name}: {e}")
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)

def check_models(config_path):
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    models_config = config["models"]
    
    image_model_path = models_config["image_generator"]["local_path"]
    eval_model_path = models_config["evaluator"]["local_path"]
    
    if not os.path.exists(image_model_path):
        raise FileNotFoundError(
            f"Image generator model not found at {image_model_path}. "
            f"Please run scripts/download_models.py first."
        )
    
    if not os.path.exists(eval_model_path):
        raise FileNotFoundError(
            f"Evaluator model not found at {eval_model_path}. "
            f"Please run scripts/download_models.py first."
        )
    
    print("All models are available.")

def generate_images(config_path):
    from scripts.generate_images import main as gen_main
    gen_main()

def create_mapping(config_path):
    pipeline = EvaluationPipeline(config_path)
    pipeline.create_image_context_mapping()

def run_comparison(config_path):
    pipeline = EvaluationPipeline(config_path)
    pipeline.run_comparison()

def evaluate_results(config_path):
    pipeline = EvaluationPipeline(config_path)
    pipeline.evaluate_results()

def find_mismatches(config_path):
    pipeline = EvaluationPipeline(config_path)
    pipeline.find_mismatch_cases()

if __name__ == "__main__":
    main()

