import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List, Dict
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
        self._load_model()
    
    def _load_model(self):
        print(f"Loading Judge model (Text-only) from {self.judge_config['local_path']}...")
        
        if not os.path.exists(self.judge_config["local_path"]):
            raise FileNotFoundError(
                f"Model not found at {self.judge_config['local_path']}. "
                f"Run scripts/download_models.py first."
            )
        
        device_map = self.judge_config.get("device_map", "auto")
        if device_map == "auto" and self.device_config["use_cuda"]:
            device_map = {"": self.device_config["cuda_device"]}
        
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
        image: Optional[any] = None, # Not used for text-only judge
        use_image: bool = False      # Not used for text-only judge
    ) -> str:
        # Ignore image and use_image arguments for text-only judge
        
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

