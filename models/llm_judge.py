import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor, GenerationConfig
from typing import Optional, List, Dict, Any
from PIL import Image
import yaml

class LLMJudge:
    def __init__(self, config_path: str = "configs/config.yaml", device_map: Optional[Any] = None):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        self.judge_config = self.config["models"].get("judge")
        if not self.judge_config:
            raise ValueError("Judge model configuration not found in config.yaml")

        self.eval_config = self.config["evaluation"]
        self.device_config = self.config["device"]
        self.device = self.device_config["cuda_device"] if self.device_config["use_cuda"] and torch.cuda.is_available() else "cpu"
        
        # If device_map is provided, update self.device to match
        if device_map is not None and isinstance(device_map, dict) and "" in device_map:
            self.device = device_map[""]

        self.torch_dtype = getattr(torch, self.judge_config["torch_dtype"])
        
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.is_vlm = False
        self._load_model(device_map)
        self.judge_generation_config = GenerationConfig(
            max_new_tokens=self.eval_config.get("judge_max_new_tokens", 1024),
            do_sample=False,
        )
    
    def _load_model(self, device_map_override: Optional[Any] = None):
        print(f"Loading Judge model from {self.judge_config['local_path']}...")
        
        if not os.path.exists(self.judge_config["local_path"]):
            raise FileNotFoundError(
                f"Model not found at {self.judge_config['local_path']}. "
                f"Run scripts/download_models.py first."
            )
        
        if device_map_override is not None:
            device_map = device_map_override
        else:
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
                local_files_only=True,
                padding_side="left"
            )
            tokenizer = getattr(self.processor, 'tokenizer', None)
            if tokenizer is not None:
                tokenizer.padding_side = 'left'
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
            if hasattr(self.processor, 'padding_side'):
                self.processor.padding_side = 'left'
            if getattr(self.processor, 'pad_token', None) is None and hasattr(self.processor, 'eos_token'):
                self.processor.pad_token = self.processor.eos_token
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
                local_files_only=True,
                padding_side="left"
            )
            self.tokenizer.padding_side = 'left'
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Judge Model loaded")
    
    def run_inference(
        self,
        prompt: str,
        image: Optional[any] = None,
        use_image: bool = False
    ) -> str:
        # Force text-only inference regardless of image input
        return self.run_batch([prompt])[0]

    def run_batch(self, prompts: List[str]) -> List[str]:
        if not prompts:
            return []
        # Since we switched to Text-only LLM (Qwen2.5-7B-Instruct), we use _run_text_batch
        return self._run_text_batch(prompts)

    def _run_text_batch(self, prompts: List[str]) -> List[str]:
        messages = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            for prompt in prompts
        ]
        texts = [
            self.tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=True
            )
            for convo in messages
        ]
        model_inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                generation_config=self.judge_generation_config,
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def _run_vlm_batch_text_only(self, prompts: List[str]) -> List[str]:
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            for prompt in prompts
        ]
        texts = [
            self.processor.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=True
            )
            for convo in messages
        ]
        inputs = self.processor(
            text=texts,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                generation_config=self.judge_generation_config,
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
    
    def _resize_image_if_needed(self, image: Image.Image, max_size: int = 1024) -> Image.Image:
        """Resize image if it exceeds the maximum dimension to prevent OOM."""
        width, height = image.size
        if max(width, height) > max_size:
            ratio = max_size / max(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            return image.resize(new_size, Image.Resampling.LANCZOS)
        return image

    def _run_vlm_inference(self, prompt: str, image: Optional[any] = None, use_image: bool = False) -> str:
        if use_image and image is not None:
            # Resize image to prevent OOM
            image = self._resize_image_if_needed(image)
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
                generation_config=self.judge_generation_config,
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

