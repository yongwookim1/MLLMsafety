import os
import json
from typing import List, Dict
import yaml


class HateCommunityLoader:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        self.dataset_path = os.path.join(
            os.path.dirname(__file__),
            "hate_community_dataset.json"
        )
        self.samples = None
        self._load_dataset()
    
    def _load_dataset(self):
        print(f"Loading hate community dataset from {self.dataset_path}...")
        
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")
        
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        
        print(f"Dataset loaded: {len(self.samples)} samples")
    
    def get_all_samples(self) -> List[Dict]:
        return self.samples
