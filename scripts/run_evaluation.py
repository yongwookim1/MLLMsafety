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
    
    print("MLLM Safety Evaluation Pipeline")
    
    pipeline = EvaluationPipeline(config_path)
    
    print()
    print("Creating image-context mapping...")
    pipeline.create_image_context_mapping()
    
    prompts = pipeline.get_prompts()
    
    for prompt in prompts:
        prompt_id = prompt.get("prompt_id")
        print()
        print(f"Running comparison (Prompt {prompt_id})...")
        pipeline.run_comparison(prompt)
        
        print()
        print(f"Evaluating results (Prompt {prompt_id})...")
        pipeline.evaluate_results(prompt)
        
        print()
        print(f"Finding mismatch cases (Prompt {prompt_id})...")
        pipeline.find_mismatch_cases(prompt)
    
    print()
    print("Pipeline completed")

if __name__ == "__main__":
    main()

