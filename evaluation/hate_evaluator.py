import os
import json
from typing import List, Dict
from PIL import Image
import yaml
import hashlib
import gc
import torch

try:
    from models.image_generator import ImageGenerator
    IMAGE_GENERATOR_AVAILABLE = True
except ImportError:
    IMAGE_GENERATOR_AVAILABLE = False
    print("Warning: ImageGenerator not available. Will use black images.")

from models.evaluator import Evaluator
from data.hate_community_loader import HateCommunityLoader
from utils.helpers import create_black_image


class HateCommunityEvaluationPipeline:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        self.evaluator = None
        self.image_generator = None
        self.data_loader = HateCommunityLoader(config_path)
        
        self.use_image_generation = IMAGE_GENERATOR_AVAILABLE
        
        self.output_config = self.config["outputs"]
        self.eval_config = self.config["evaluation"]
        self.image_output_dir = self.config.get("image_generation", {}).get("output_dir", "./outputs/hate_images")
        
        self.shared_image_path = self.config.get("image_generation", {}).get("shared_image_path", None)
        
        os.makedirs(self.eval_config["output_dir"], exist_ok=True)
        os.makedirs(self.image_output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.output_config["comparison_results"]), exist_ok=True)
    
    def _load_image_generator(self):
        if self.image_generator is None and self.use_image_generation:
            try:
                print("\n" + "=" * 60)
                print("Loading ImageGenerator (Qwen-Image)...")
                print("=" * 60)
                self.image_generator = ImageGenerator(self.config_path)
                print("ImageGenerator loaded successfully!")
            except Exception as e:
                print(f"Warning: Could not load ImageGenerator: {e}")
                print("Will use black images instead.")
                self.image_generator = None
                self.use_image_generation = False
    
    def _unload_image_generator(self):
        if self.image_generator is not None:
            print("\n" + "=" * 60)
            print("Unloading ImageGenerator to free memory...")
            print("=" * 60)
            del self.image_generator
            self.image_generator = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("ImageGenerator unloaded.")
    
    def _load_evaluator(self):
        if self.evaluator is None:
            print("\n" + "=" * 60)
            print("Loading Evaluator (Qwen2.5-VL)...")
            print("=" * 60)
            self.evaluator = Evaluator(self.config_path)
            print("Evaluator loaded successfully!")
    
    def _unload_evaluator(self):
        if self.evaluator is not None:
            print("\n" + "=" * 60)
            print("Unloading Evaluator to free memory...")
            print("=" * 60)
            del self.evaluator
            self.evaluator = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Evaluator unloaded.")
    
    def _get_image_path(self, question: str, sample_id: str) -> str:
        if self.shared_image_path and os.path.exists(self.shared_image_path):
            return self.shared_image_path
        
        question_hash = hashlib.md5(question.encode()).hexdigest()[:8]
        safe_sample_id = sample_id.replace("/", "_").replace("\\", "_")
        image_filename = f"{safe_sample_id}_{question_hash}.jpg"
        return os.path.join(self.image_output_dir, image_filename)
    
    def generate_all_images(self, max_samples: int = None) -> Dict[str, str]:
        print("\n" + "=" * 60)
        print("STEP 1: Generating Images")
        print("=" * 60)
        
        if self.shared_image_path and os.path.exists(self.shared_image_path):
            print(f"Using shared image: {self.shared_image_path}")
            samples = self.data_loader.get_all_samples()
            if max_samples is not None:
                samples = samples[:max_samples]
            image_paths = {sample.get("sample_id", f"sample_{idx}"): self.shared_image_path 
                          for idx, sample in enumerate(samples)}
            print(f"All {len(image_paths)} samples will use the shared image.")
            return image_paths
        
        self._load_image_generator()
        
        samples = self.data_loader.get_all_samples()
        if max_samples is not None:
            samples = samples[:max_samples]
        image_paths = {}
        total_samples = len(samples)
        
        print(f"Total samples to process: {total_samples}")
        
        for idx, sample in enumerate(samples):
            sample_id = sample.get("sample_id", f"sample_{idx}")
            question = sample.get("question", "")
            
            if not question:
                continue
            
            image_path = self._get_image_path(question, sample_id)
            image_paths[sample_id] = image_path
            
            if os.path.exists(image_path):
                print(f"[{idx + 1}/{total_samples}] Image already exists: {sample_id}")
                continue
            
            if self.use_image_generation and self.image_generator:
                try:
                    print(f"[{idx + 1}/{total_samples}] Generating image for: {question[:50]}...")
                    image = self.image_generator.generate(
                        prompt=question,
                        negative_prompt=" ",
                        seed=42
                    )
                    image.save(image_path)
                    print(f"  Image saved to: {image_path}")
                except Exception as e:
                    print(f"  Error generating image: {e}")
                    print(f"  Will use black image for evaluation")
            else:
                black_image = create_black_image(width=512, height=512)
                black_image.save(image_path)
                print(f"[{idx + 1}/{total_samples}] Created black image: {sample_id}")
        
        self._unload_image_generator()
        
        print(f"\nImage generation complete! Generated {len(image_paths)} images.")
        return image_paths
    
    def run_evaluation(self, image_paths: Dict[str, str] = None, max_samples: int = None) -> List[Dict]:
        print("\n" + "=" * 60)
        print("STEP 2: Running Evaluation")
        print("=" * 60)
        
        self._load_evaluator()
        
        samples = self.data_loader.get_all_samples()
        if max_samples is not None:
            samples = samples[:max_samples]
        results = []
        results_file = self.output_config.get("hate_community_results", "./outputs/hate_community_results.json")
        
        total_samples = len(samples)
        print(f"Total samples to evaluate: {total_samples}")
        print("Each question will be evaluated with and without image.")
        
        for idx, sample in enumerate(samples):
            sample_id = sample.get("sample_id", f"sample_{idx}")
            question = sample.get("question", "")
            
            if not question:
                continue
            
            text_only_prompt = f"Answer the following question.\n\nQuestion: {question}"
            multimodal_prompt = f"Look at the image and answer the following question.\n\nQuestion: {question}"
            
            print(f"\n[{idx + 1}/{total_samples}] Sample ID: {sample_id}")
            print(f"Question: {question}")
            
            try:
                if image_paths and sample_id in image_paths:
                    image_path = image_paths[sample_id]
                    if os.path.exists(image_path):
                        image = Image.open(image_path).convert("RGB")
                    else:
                        image = create_black_image(width=512, height=512)
                else:
                    image_path = self._get_image_path(question, sample_id)
                    if os.path.exists(image_path):
                        image = Image.open(image_path).convert("RGB")
                    else:
                        image = create_black_image(width=512, height=512)
                
                text_only_response = self.evaluator.run_inference(
                    text_only_prompt,
                    image=None,
                    use_image=False
                )
                
                multimodal_response = self.evaluator.run_inference(
                    multimodal_prompt,
                    image=image,
                    use_image=True
                )
                
                result = {
                    "sample_id": sample_id,
                    "question": question,
                    "text_only_response": text_only_response,
                    "multimodal_response": multimodal_response
                }
                
                results.append(result)
                
                if (idx + 1) % self.eval_config.get("save_interval", 10) == 0:
                    with open(results_file, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    print(f"Saved {idx + 1}/{total_samples} results")
                
                print(f"Text-only Response: {text_only_response}")
                print(f"Multimodal Response: {multimodal_response}")
                print("-" * 60)
            
            except Exception as e:
                print(f"Error processing sample {sample_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\nEvaluation complete! Results saved to {results_file}")
        return results
    
    def run_full_pipeline(self, max_samples: int = None) -> List[Dict]:
        print("\n" + "=" * 60)
        print("Hate Community Evaluation Pipeline")
        print("(Sequential Loading: Images → Evaluation)")
        print("=" * 60)
        
        if max_samples is not None:
            print(f"Limiting to first {max_samples} samples")
        
        image_paths = self.generate_all_images(max_samples=max_samples)
        results = self.run_evaluation(image_paths, max_samples=max_samples)
        
        return results
    
    def print_summary(self, results: List[Dict]):
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        
        total_samples = len(results)
        
        print(f"\nTotal Samples: {total_samples}")
        print("Each sample was evaluated with:")
        print("  - Text-only mode (no image)")
        print("  - Multimodal mode (with image)")
        
        print("\n" + "=" * 60)
        print("Sample Results (First 10)")
        print("=" * 60)
        
        for idx, result in enumerate(results[:10]):
            print(f"\n[{idx + 1}] Sample ID: {result.get('sample_id', 'N/A')}")
            print(f"Question: {result.get('question', 'N/A')}")
            print(f"Text-only: {result.get('text_only_response', 'N/A')[:100]}...")
            print(f"Multimodal: {result.get('multimodal_response', 'N/A')[:100]}...")
        
        print("\n" + "=" * 60)
