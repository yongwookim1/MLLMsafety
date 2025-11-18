import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from evaluation import HateCommunityEvaluationPipeline

def main():
    config_path = "configs/config.yaml"
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("Hate Community Evaluation Pipeline")
    print("(Sequential Loading: Images → Evaluation)")
    print("=" * 60)
    
    pipeline = HateCommunityEvaluationPipeline(config_path)
    
    max_samples = 21
    
    print("\nRunning full pipeline (images first, then evaluation)...")
    print(f"Processing first {max_samples} samples only")
    results = pipeline.run_full_pipeline(max_samples=max_samples)
    
    print("\nGenerating summary...")
    pipeline.print_summary(results)
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()

