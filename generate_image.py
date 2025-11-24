import os
import torch
import hashlib
from diffusers import DiffusionPipeline
from PIL import Image
from typing import Optional


class QwenImageGenerator:
    def __init__(self, model_path: str = "./models_cache/qwen-image", use_cuda: bool = True, use_memory_efficient: bool = False):
        self.model_path = os.path.abspath(model_path)
        
        self.device = self.get_empty_device() if use_cuda else "cpu"
        
        self.torch_dtype = torch.bfloat16
        self.use_memory_efficient = use_memory_efficient
        self.pipeline = None
        self._load_model()

    def get_empty_device(self):
        if not torch.cuda.is_available(): return "cpu"
        try:
            frees = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
            best_id = frees.index(max(frees))
            print(f"Using GPU: cuda:{best_id} ({max(frees)/1024**3:.1f}GB free)")
            return f"cuda:{best_id}"
        except Exception as e:
            print(f"GPU select failed: {e}, using cuda:0")
            return "cuda:0"
    
    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                f"Run scripts/download_models.py first."
            )
        
        print(f"Loading model from {self.model_path}...")
        
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            local_files_only=True
        )
        
        if self.use_memory_efficient:
            print("Enabling memory-efficient mode...")
            self.pipeline.enable_model_cpu_offload()
            self.pipeline.enable_vae_slicing()
            self.pipeline.enable_attention_slicing(2)
            if hasattr(self.pipeline, 'enable_vae_tiling'):
                self.pipeline.enable_vae_tiling()
        else:
            self.pipeline = self.pipeline.to(self.device)
            self.pipeline.enable_attention_slicing(2)
        
        print("Model loaded")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = " ",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 28,
        true_cfg_scale: float = 4.0,
        seed: Optional[int] = None
    ) -> Image.Image:
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
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale,
                generator=generator
            ).images[0]
        
        return image
    
    def generate_if_new(
        self,
        prompt: str,
        output_dir: str,
        filename_prefix: str = "",
        **kwargs
    ) -> Optional[str]:
        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        
        if filename_prefix:
            filename = f"{filename_prefix}_{prompt_hash}.jpg"
        else:
            filename = f"{prompt_hash}.jpg"
            
        output_path = os.path.join(output_dir, filename)
        
        if os.path.exists(output_path):
            print(f"Skipping: {prompt[:30]}... (exists)")
            return None
            
        print(f"Generating: {prompt[:30]}...")
        image = self.generate(prompt, **kwargs)
        
        os.makedirs(output_dir, exist_ok=True)
        self.save_image(image, output_path)
        return output_path
    
    def save_image(self, image: Image.Image, output_path: str):
        image.save(output_path)
        print(f"Saved: {output_path}")


def main():
    generator = QwenImageGenerator()
    
    prompt = "지난주 이마트 앞에서 카카오택시로 택시를 잡으려는 손자와 할머니를 봤습니다."
    output_dir = "./outputs"
    
    output_path = generator.generate_if_new(prompt, output_dir, filename_prefix="demo")
    
    if output_path:
        print(f"Done: {output_path}")
    else:
        print("Skipped (already exists)")


if __name__ == "__main__":
    main()
