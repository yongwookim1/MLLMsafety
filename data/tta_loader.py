import os
from typing import Dict, List, Any, Optional
from datasets import load_from_disk
import yaml

class TTALoader:
    def __init__(self, config_path: str = "configs/config.yaml", split: str = "eval"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        # Modified to load from local disk
        # Using config value if present, otherwise defaulting to relative path
        self.local_cache_dir = self.config.get("dataset", {}).get("local_cache_dir", "./data_cache")
        self.dataset_path = os.path.join(self.local_cache_dir, "TTA01_AssurAI")
        
        self.split = split
        self.dataset = None
        self._load_dataset()

    def _load_dataset(self):
        print(f"Loading AssurAI dataset from {self.dataset_path}...")
        try:
            if not os.path.exists(self.dataset_path):
                raise FileNotFoundError(f"Dataset not found at {self.dataset_path}. Please run scripts/download_tta_dataset.py first and ensure the folder is transferred.")
                
            self.dataset = load_from_disk(self.dataset_path)
            print(f"Loaded {len(self.dataset)} samples.")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def get_text_samples(self) -> List[Dict[str, Any]]:
        """
        Filter and return samples where modality is 'text'.
        These are the candidates for multimodal expansion.
        """
        if self.dataset is None:
            return []
            
        # Filter for text modality
        text_samples = [
            item for item in self.dataset 
            if item.get('modality') == 'text'
        ]
        return text_samples

    def get_image_samples(self) -> List[Dict[str, Any]]:
        """
        Filter and return samples where modality is 'image'.
        These prompts are explicitly requesting image generation.
        """
        if self.dataset is None:
            return []
            
        image_samples = [
            item for item in self.dataset 
            if item.get('modality') == 'image'
        ]
        return image_samples

    def get_all_samples(self) -> List[Dict[str, Any]]:
        if self.dataset is None:
            return []
        return list(self.dataset)

    def get_sample_by_id(self, sample_id: str) -> Optional[Dict[str, Any]]:
        if self.dataset is None:
            return None
        
        for item in self.dataset:
            if item.get('id') == sample_id:
                return item
        return None
