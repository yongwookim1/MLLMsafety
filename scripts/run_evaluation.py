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
    
    print("=" * 60)
    print("MLLM Safety Evaluation Pipeline")
    print("(Text-only vs Multimodal Comparison)")
    print("=" * 60)
    
    pipeline = EvaluationPipeline(config_path)
    
    print("\nStep 1: Creating image-context mapping...")
    pipeline.create_image_context_mapping()
    
    print("\nStep 2: Running comparison (Text-only vs Multimodal)...")
    pipeline.run_comparison()
    
    print("\nStep 3: Evaluating results...")
    pipeline.evaluate_results()
    
    print("\nStep 4: Finding mismatch cases...")
    pipeline.find_mismatch_cases()
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()

