import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from typing import Optional
from PIL import Image
import yaml


class Evaluator:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config["models"]["evaluator"]
        self.eval_config = self.config["evaluation"]
        self.device_config = self.config["device"]
        
        self.device = self.device_config["cuda_device"] if self.device_config["use_cuda"] and torch.cuda.is_available() else "cpu"
        self.torch_dtype = getattr(torch, self.model_config["torch_dtype"])
        
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        print(f"Loading Qwen2.5-VL model from {self.model_config['local_path']}...")
        
        if not os.path.exists(self.model_config["local_path"]):
            raise FileNotFoundError(
                f"Model not found at {self.model_config['local_path']}. "
                f"Run scripts/download_models.py first."
            )
        
        device_map = self.model_config.get("device_map", "auto")
        if device_map == "auto" and self.device_config["use_cuda"]:
            device_map = {"": self.device_config["cuda_device"]}
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_config["local_path"],
            torch_dtype=self.torch_dtype,
            device_map=device_map,
            local_files_only=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_config["local_path"],
            local_files_only=True
        )
        
        print("Model loaded")
    
    def run_inference(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        use_image: bool = True
    ) -> str:
        if use_image and image is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        if use_image and image is not None:
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            )
        else:
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt"
            )
        
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.eval_config["max_new_tokens"],
                temperature=self.eval_config.get("temperature", 0.0),
                do_sample=False,
                top_p=None,
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return response
