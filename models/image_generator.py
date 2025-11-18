import os
import torch
from diffusers import DiffusionPipeline
from typing import Optional
import yaml


class ImageGenerator:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config["models"]["image_generator"]
        self.gen_config = self.config["image_generation"]
        self.device_config = self.config["device"]
        
        self.device = self.device_config["cuda_device"] if self.device_config["use_cuda"] and torch.cuda.is_available() else "cpu"
        self.torch_dtype = getattr(torch, self.model_config["torch_dtype"])
        
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        print(f"Loading Qwen-Image model from {self.model_config['local_path']}...")
        
        if not os.path.exists(self.model_config["local_path"]):
            raise FileNotFoundError(
                f"Model not found at {self.model_config['local_path']}. "
                f"Please run scripts/download_models.py first to download the model."
            )
        
        use_memory_efficient = self.model_config.get("use_memory_efficient", False)
        
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.model_config["local_path"],
            torch_dtype=self.torch_dtype,
            local_files_only=True
        )
        
        if use_memory_efficient:
            print("Enabling memory-efficient optimizations (slower but uses less VRAM)...")
            self.pipeline.enable_model_cpu_offload()
            self.pipeline.enable_vae_slicing()
            self.pipeline.enable_attention_slicing(2)
            if hasattr(self.pipeline, 'enable_vae_tiling'):
                self.pipeline.enable_vae_tiling()
            print("Memory optimizations enabled!")
        else:
            print("Loading model to GPU for maximum speed...")
            self.pipeline = self.pipeline.to(self.device)
            self.pipeline.enable_attention_slicing(2)
        
        print("Model loaded successfully!")
    
    def generate(self, prompt: str, negative_prompt: str = " ", seed: Optional[int] = None):
        base_width = self.gen_config.get("width", 512)
        base_height = self.gen_config.get("height", 512)
        
        aspect_ratios = {
            "1:1": (base_width, base_height),
            "16:9": (int(base_width * 1.78), base_height),
            "9:16": (base_width, int(base_height * 1.78)),
            "4:3": (int(base_width * 1.33), base_height),
            "3:4": (base_width, int(base_height * 1.33)),
            "3:2": (int(base_width * 1.5), base_height),
            "2:3": (base_width, int(base_height * 1.5)),
        }
        
        aspect_ratio = self.gen_config["aspect_ratio"]
        width, height = aspect_ratios.get(aspect_ratio, (base_width, base_height))
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        device_type = "cuda" if "cuda" in self.device else "cpu"
        with torch.autocast(device_type=device_type, dtype=self.torch_dtype):
            image = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=self.gen_config["num_inference_steps"],
                true_cfg_scale=self.gen_config["true_cfg_scale"],
                generator=generator
            ).images[0]
        
        return image
