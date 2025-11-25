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
            
            # Use existing category or infer from ID
            category = sample.get("bbq_category")
            if not category or category == "Unknown":
                category = self.infer_category_from_id(sample.get("sample_id", ""))
            
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

    def get_all_unique_samples(self, max_samples: int = None, categories: List[str] = None,
                              bias_types: List[str] = None) -> List[Dict]:
        """
        Get all unique samples, optionally filtered by categories and bias types.
        Unlike get_unique_contexts, this returns all samples, not just unique contexts.
        """
        samples = self.get_all_samples()

        # Filter by categories if specified
        if categories:
            samples = [s for s in samples if s.get("bbq_category") in categories]

        # Filter by bias types if specified
        if bias_types:
            samples = [s for s in samples if s.get("bias_type") in bias_types]

        # Limit samples if specified
        if max_samples:
            samples = samples[:max_samples]

        # Convert to the format expected by generate_images.py
        result = []
        seen_contexts = set()

        for sample in samples:
            context = sample.get("context", "")
            if context and context.strip() and context not in seen_contexts:
                seen_contexts.add(context)
                result.append({
                    "context": context,
                    "sample_id": sample.get("sample_id", ""),
                    "bbq_category": sample.get("bbq_category", ""),
                    "bias_type": sample.get("bias_type", "")
                })

        return result

    def get_samples_by_category(self, category: str, max_samples: int = None) -> List[Dict]:
        """Get samples for a specific category"""
        return self.get_all_unique_samples(categories=[category], max_samples=max_samples)

    def _parse_sample_id(self, sample_id: str) -> Dict[str, str]:
        parts = sample_id.split("-")
        context_type = "ambiguous" if "amb" in sample_id else "disambiguated"
        bias_type = "biased" if "bsd" in sample_id else "counter_biased"
        return {
            "context_type": context_type,
            "bias_type": bias_type
        }
    
    def infer_category_from_id(self, sample_id: str) -> str:
        """Infer BBQ category from sample ID prefix if missing"""
        if not sample_id:
            return "Unknown"
            
        prefix_map = {
            "age": "Age",
            "disability_status": "Disability_status",
            "gender_identity": "Gender_identity",
            "nationality": "Nationality",
            "physical_appearance": "Physical_appearance",
            "race_ethnicity": "Race_ethnicity",
            "race_ethnicity_nationality": "Race_ethnicity",
            "religion": "Religion",
            "ses": "SES",
            "sexual_orientation": "Sexual_orientation",
            "domestic_area_of_origin": "Domestic Area of Origin",
            "family_structure": "Family Structure",
            "political_orientation": "Political Orientation",
            "educational_background": "Education Background"
        }
        
        # Check for exact prefix match first (e.g., "age-...")
        for prefix, category in prefix_map.items():
            if sample_id.startswith(prefix + "-"):
                return category
                
        return "Unknown"

    def get_all_samples(self) -> List[Dict]:
        samples = []
        for sample in self.dataset:
            sample_id = sample.get("sample_id", "")
            metadata = self._parse_sample_id(sample_id)
            
            # Use existing category or infer from ID
            category = sample.get("bbq_category")
            if not category or category == "Unknown":
                category = self.infer_category_from_id(sample_id)
                
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
                "bbq_category": category,
                "label_annotation": sample.get("label_annotation", ""),
                "context_type": metadata["context_type"],
                "bias_type": metadata["bias_type"]
            })
        return samples

