import os
from datasets import load_from_disk
from typing import List, Dict
import yaml
import ast


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
                f"Run 'python scripts/download_dataset.py' first."
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
        
        from collections import defaultdict
        category_contexts = defaultdict(list)
        
        for sample in self.dataset:
            context = sample.get("context", "")
            category = sample.get("bbq_category", "Unknown")
            
            if context and context.strip() and context not in seen_contexts:
                seen_contexts.add(context)
                category_contexts[category].append({
                    "context": context,
                    "sample_id": sample.get("sample_id", "")
                })
        
        contexts_per_category = max_contexts // len(category_contexts) if category_contexts else max_contexts
        remaining = max_contexts % len(category_contexts) if category_contexts else 0
        
        for category, cat_contexts in category_contexts.items():
            take_count = min(len(cat_contexts), contexts_per_category + (1 if remaining > 0 else 0))
            if remaining > 0:
                remaining -= 1
            contexts.extend(cat_contexts[:take_count])
        
        if len(contexts) < max_contexts:
            seen_in_balanced = set(c["context"] for c in contexts)
            for sample in self.dataset:
                if len(contexts) >= max_contexts:
                    break
                context = sample.get("context", "")
                if context and context.strip() and context not in seen_in_balanced:
                    seen_in_balanced.add(context)
                    contexts.append({
                        "context": context,
                        "sample_id": sample.get("sample_id", "")
                    })
        
        return contexts
    
    def _parse_sample_id(self, sample_id: str) -> Dict[str, str]:
        parts = sample_id.split("-")
        context_type = "ambiguous" if "amb" in sample_id else "disambiguated"
        bias_type = "biased" if "bsd" in sample_id else "counter_biased"
        return {
            "context_type": context_type,
            "bias_type": bias_type
        }
    
    def get_all_samples(self) -> List[Dict]:
        samples = []
        for sample in self.dataset:
            sample_id = sample.get("sample_id", "")
            metadata = self._parse_sample_id(sample_id)
            choices = sample.get("choices", [])
            if isinstance(choices, str):
                try:
                    choices = ast.literal_eval(choices)
                except:
                    choices = []
            samples.append({
                "sample_id": sample_id,
                "context": sample.get("context", ""),
                "question": sample.get("question", ""),
                "choices": choices,
                "answer": sample.get("answer", ""),
                "biased_answer": sample.get("biased_answer", ""),
                "bbq_category": sample.get("bbq_category", ""),
                "label_annotation": sample.get("label_annotation", ""),
                "context_type": metadata["context_type"],
                "bias_type": metadata["bias_type"]
            })
        return samples

