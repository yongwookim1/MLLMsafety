import os
import torch
from diffusers import DiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
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
        
        self.model_type = self.model_config.get("type", "qwen-image")
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        if self.model_type == "kimchi":
            self._load_kimchi_model()
        else:
            self._load_qwen_model()
    
    def _load_qwen_model(self):
        print(f"Loading Qwen-Image model from {self.model_config['local_path']}...")
        
        if not os.path.exists(self.model_config["local_path"]):
            raise FileNotFoundError(
                f"Model not found at {self.model_config['local_path']}. "
                f"Run scripts/download_models.py first."
            )
        
        use_memory_efficient = self.model_config.get("use_memory_efficient", False)
        
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.model_config["local_path"],
            torch_dtype=self.torch_dtype,
            local_files_only=True
        )
        
        if use_memory_efficient:
            print("Enabling aggressive memory-efficient mode...")
            # Clear any existing GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.pipeline.enable_model_cpu_offload()
            self.pipeline.enable_vae_slicing()
            self.pipeline.enable_attention_slicing(8)  # More aggressive slicing

            # Move VAE to CPU explicitly to save GPU memory
            if hasattr(self.pipeline, 'vae'):
                self.pipeline.vae.to('cpu')
                print("Moved VAE to CPU")

            if hasattr(self.pipeline, 'enable_vae_tiling'):
                self.pipeline.enable_vae_tiling()

            # Skip xFormers for now due to compatibility issues
            # try:
            #     self.pipeline.enable_xformers_memory_efficient_attention()
            #     print("Enabled xFormers memory efficient attention")
            # except Exception as e:
            #     print(f"xFormers not available: {e}")
            print("Using standard attention for compatibility")

            # Additional memory settings
            torch.cuda.set_per_process_memory_fraction(0.9)  # Use up to 90% of GPU memory
        else:
            self.pipeline = self.pipeline.to(self.device)
            self.pipeline.enable_attention_slicing(2)
        
        print("Model loaded")
    
    def _load_kimchi_model(self):
        print("Loading Kimchi model (Stable Diffusion + Korean CLIP + LoRA)...")
        
        pretrained_model_path = self.model_config.get("pretrained_model_path", "./models_cache/stable-diffusion-v1-5")
        lora_path = self.model_config.get("lora_path")
        clip_model_path = self.model_config.get("clip_model_path", "./models_cache/clip-vit-large-patch14-ko")
        
        if not os.path.exists(pretrained_model_path) or not os.path.isdir(pretrained_model_path):
            raise FileNotFoundError(f"Pretrained model not found at {pretrained_model_path}")
        
        if not os.path.exists(clip_model_path) or not os.path.isdir(clip_model_path):
            raise FileNotFoundError(f"CLIP model not found at {clip_model_path}")
        
        if lora_path and not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA weights not found at {lora_path}")
        
        print(f"Loading Stable Diffusion from {pretrained_model_path}...")
        self.pipeline = DiffusionPipeline.from_pretrained(
            pretrained_model_path,
            torch_dtype=self.torch_dtype,
            local_files_only=True
        )
        
        print(f"Loading Korean CLIP model from {clip_model_path}...")
        text_encoder = CLIPTextModel.from_pretrained(
            clip_model_path,
            local_files_only=True
        )
        text_encoder.to(self.device)
        tokenizer = CLIPTokenizer.from_pretrained(
            clip_model_path,
            local_files_only=True
        )
        
        self.pipeline.text_encoder = text_encoder
        self.pipeline.tokenizer = tokenizer
        
        if lora_path:
            print(f"Loading LoRA weights from {lora_path}...")
            self.pipeline.load_lora_weights(lora_path)
        
        self.pipeline = self.pipeline.to(self.device)
        self.pipeline.enable_attention_slicing(2)
        
        print("Kimchi model loaded")
    
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

        try:
            if self.model_type == "kimchi":
                with torch.autocast(device_type=device_type):
                    result = self.pipeline(
                        prompt,
                        num_inference_steps=self.gen_config["num_inference_steps"]
                    )
                    if hasattr(result, 'images') and result.images:
                        image = result.images[0]
                    else:
                        raise ValueError("Pipeline returned no images")
            else:
                with torch.autocast(device_type=device_type, dtype=self.torch_dtype):
                    result = self.pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        num_inference_steps=self.gen_config["num_inference_steps"],
                        true_cfg_scale=self.gen_config["true_cfg_scale"],
                        generator=generator
                    )
                    if hasattr(result, 'images') and len(result.images) > 0:
                        image = result.images[0]
                    else:
                        raise ValueError(f"Pipeline returned no images. Result type: {type(result)}, has images: {hasattr(result, 'images')}")

            return image

        except Exception as e:
            print(f"Error during image generation: {e}")
            # Clear GPU memory on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
