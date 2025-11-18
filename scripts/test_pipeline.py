import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

def test_imports():
    print("Testing imports...")
    try:
        from models import ImageGenerator, Evaluator
        from data import KOBBQLoader
        from evaluation import EvaluationPipeline
        from utils import extract_answer, format_choices, create_black_image
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_config():
    print("\nTesting config file...")
    try:
        import yaml
        config_path = "configs/config.yaml"
        if not os.path.exists(config_path):
            print(f"✗ Config file not found: {config_path}")
            return False
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        required_keys = ["models", "dataset", "image_generation", "evaluation", "device", "outputs"]
        for key in required_keys:
            if key not in config:
                print(f"✗ Missing config key: {key}")
                return False
        
        print("✓ Config file is valid")
        return True
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False

def test_paths():
    print("\nTesting paths...")
    try:
        config_path = "configs/config.yaml"
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        paths_to_check = [
            config["models"]["image_generator"]["local_path"],
            config["models"]["evaluator"]["local_path"],
            config["image_generation"]["output_dir"],
            config["dataset"]["local_cache_dir"]
        ]
        
        for path in paths_to_check:
            if not os.path.isabs(path):
                abs_path = os.path.abspath(path)
                print(f"  {path} -> {abs_path}")
        
        print("✓ Paths are valid")
        return True
    except Exception as e:
        print(f"✗ Path error: {e}")
        return False

def test_data_loader():
    print("\nTesting data loader initialization...")
    try:
        from data import KOBBQLoader
        loader = KOBBQLoader()
        print(f"✓ Data loader initialized (dataset size: {len(loader.dataset)})")
        return True
    except Exception as e:
        print(f"✗ Data loader error: {e}")
        print("  Note: This might require internet connection for first-time dataset download")
        return False

def main():
    print("=" * 60)
    print("Pipeline Test Suite")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Config", test_config),
        ("Paths", test_paths),
        ("Data Loader", test_data_loader)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

