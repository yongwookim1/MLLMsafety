import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from evaluation import EvaluationPipeline

def main():
    config_path = "configs/config.yaml"
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    
    print("Full MLLM Safety Evaluation Pipeline")
    
    check_models(config_path)
    generate_images(config_path)
    create_mapping(config_path)
    
    pipeline = EvaluationPipeline(config_path)
    prompts = pipeline.get_prompts()
    for prompt in prompts:
        print()
        print(f"Processing prompt {prompt.get('prompt_id')}...")
        try:
            pipeline.run_comparison(prompt)
            pipeline.evaluate_results(prompt)
            pipeline.find_mismatch_cases(prompt)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    print()
    print("Pipeline completed")

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
            f"Run scripts/download_models.py first."
        )
    
    if not os.path.exists(eval_model_path):
        raise FileNotFoundError(
            f"Evaluator model not found at {eval_model_path}. "
            f"Run scripts/download_models.py first."
        )
    
    print("Models available")

def generate_images(config_path):
    from scripts.generate_images import main as gen_main
    gen_main()

def create_mapping(config_path):
    pipeline = EvaluationPipeline(config_path)
    pipeline.create_image_context_mapping()

if __name__ == "__main__":
    main()

