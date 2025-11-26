import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor, GenerationConfig
from typing import Optional, List, Dict
from PIL import Image
import yaml

class LLMJudge:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        self.judge_config = self.config["models"].get("judge")
        if not self.judge_config:
            raise ValueError("Judge model configuration not found in config.yaml")

        self.eval_config = self.config["evaluation"]
        self.device_config = self.config["device"]
        self.device = self.device_config["cuda_device"] if self.device_config["use_cuda"] and torch.cuda.is_available() else "cpu"
        self.torch_dtype = getattr(torch, self.judge_config["torch_dtype"])
        
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.is_vlm = False
        self._load_model()
    
    def _load_model(self):
        print(f"Loading Judge model from {self.judge_config['local_path']}...")
        
        if not os.path.exists(self.judge_config["local_path"]):
            raise FileNotFoundError(
                f"Model not found at {self.judge_config['local_path']}. "
                f"Run scripts/download_models.py first."
            )
        
        device_map = self.judge_config.get("device_map", "auto")
        if device_map == "auto" and self.device_config["use_cuda"]:
            device_map = {"": self.device_config["cuda_device"]}
            
        # Check if model is VLM based on name or config
        self.is_vlm = "VL" in self.judge_config.get("name", "") or "VL" in self.judge_config.get("local_path", "")
        
        if self.is_vlm:
            print("Detected VLM model. Loading with Qwen2_5_VLForConditionalGeneration...")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.judge_config["local_path"],
                torch_dtype=self.torch_dtype,
                device_map=device_map,
                local_files_only=True
            )
            self.processor = AutoProcessor.from_pretrained(
                self.judge_config["local_path"],
                local_files_only=True
            )
        else:
            print("Loading Text-only model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.judge_config["local_path"],
                torch_dtype=self.torch_dtype,
                device_map=device_map,
                local_files_only=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.judge_config["local_path"],
                local_files_only=True
            )
        
        print("Judge Model loaded")
    
    def run_inference(
        self,
        prompt: str,
        image: Optional[any] = None,
        use_image: bool = False
    ) -> str:
        if self.is_vlm:
            return self._run_vlm_inference(prompt, image, use_image)
        
        # Text-only inference
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=1024,
                do_sample=False
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def _run_vlm_inference(self, prompt: str, image: Optional[any] = None, use_image: bool = False) -> str:
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
            generation_config = GenerationConfig(
                max_new_tokens=1024,
                do_sample=False,
            )
            
            generated_ids = self.model.generate(
                **inputs,
                generation_config=generation_config,
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

