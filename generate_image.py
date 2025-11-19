import os
import torch
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
            print(f"Auto-selected GPU: cuda:{best_id} (Free: {max(frees)/1024**3:.1f}GB)")
            return f"cuda:{best_id}"
        except Exception as e:
            print(f"GPU auto-select failed: {e}. Defaulting to cuda:0")
            return "cuda:0"
    
    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                f"Please run scripts/download_models.py first."
            )
        
        print(f"Loading Qwen-Image model from {self.model_path}...")
        
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            local_files_only=True
        )
        
        if self.use_memory_efficient:
            print("Enabling memory-efficient optimizations (slower but uses less VRAM)...")
            self.pipeline.enable_model_cpu_offload()
            self.pipeline.enable_vae_slicing()
            self.pipeline.enable_attention_slicing(2)
            if hasattr(self.pipeline, 'enable_vae_tiling'):
                self.pipeline.enable_vae_tiling()
        else:
            print(f"Loading model to {self.device} for maximum speed...")
            self.pipeline = self.pipeline.to(self.device)
            self.pipeline.enable_attention_slicing(2)
        
        print("Model loaded successfully!")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = " ",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 50,
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
    
    def save_image(self, image: Image.Image, output_path: str):
        image.save(output_path)
        print(f"Image saved to {output_path}")


def main():
    generator = QwenImageGenerator()
    
    prompt = "지난주 이마트 앞에서 카카오택시로 택시를 잡으려는 손자와 할머니를 봤습니다."
    image = generator.generate(prompt)
    
    output_path = "./outputs/generated_image.jpg"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    generator.save_image(image, output_path)
    
    print(f"Generated image from prompt: '{prompt}'")


if __name__ == "__main__":
    main()
