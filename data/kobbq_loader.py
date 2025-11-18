import os
from datasets import load_from_disk
from typing import List, Dict
import yaml


class KOBBQLoader:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        self.dataset_config = self.config["dataset"]
        self.cache_dir = self.dataset_config.get("local_cache_dir", "./data_cache")
        self.dataset = None
        self._load_dataset()
    
    def _load_dataset(self):
        print(f"Loading KOBBQ dataset from {self.dataset_config['name']}...")
        
        cache_dir_abs = os.path.abspath(self.cache_dir)
        dataset_name = self.dataset_config["name"]
        dataset_split = self.dataset_config["split"]
        dataset_path = os.path.join(cache_dir_abs, dataset_name.replace("/", "_"), dataset_split)
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Dataset not found at {dataset_path}. "
                f"Please run 'python scripts/download_dataset.py' first to download and save the dataset."
            )
        
        try:
            self.dataset = load_from_disk(dataset_path)
            print(f"Dataset loaded: {len(self.dataset)} samples")
        except Exception as e:
            print(f"Warning: Failed to load dataset: {e}")
            raise
    
    def get_unique_contexts(self, max_contexts: int = 10000) -> List[Dict]:
        seen_contexts = set()
        contexts = []
        
        for sample in self.dataset:
            if len(contexts) >= max_contexts:
                break
            
            context = sample.get("context", "")
            if context and context.strip() and context not in seen_contexts:
                seen_contexts.add(context)
                contexts.append({
                    "context": context,
                    "sample_id": sample.get("sample_id", "")
                })
        
        return contexts
    
    def get_all_samples(self) -> List[Dict]:
        samples = []
        for sample in self.dataset:
            samples.append({
                "sample_id": sample.get("sample_id", ""),
                "context": sample.get("context", ""),
                "question": sample.get("question", ""),
                "choices": sample.get("choices", []),
                "answer": sample.get("answer", ""),
                "biased_answer": sample.get("biased_answer", ""),
                "bbq_category": sample.get("bbq_category", "")
            })
        return samples

