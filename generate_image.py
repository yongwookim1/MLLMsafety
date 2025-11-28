import os
import torch
import hashlib
from diffusers import DiffusionPipeline
from PIL import Image
from typing import Optional


class QwenImageGenerator:
    """
    A wrapper class for generating images using Qwen's diffusion model.

    This class handles model loading, device selection, and provides a clean interface
    for generating images from text prompts with optional memory efficiency settings.
    """

    def __init__(self, model_path: str = "./models_cache/qwen-image", use_cuda: bool = True, use_memory_efficient: bool = False):
        """
        Initialize the image generator.

        Args:
            model_path: Path to the pre-trained model directory
            use_cuda: Whether to use CUDA if available
            use_memory_efficient: Whether to enable memory-saving optimizations
        """
        self.model_path = os.path.abspath(model_path)
        self.use_memory_efficient = use_memory_efficient

        # Set up device and data type
        self.device = self._select_best_device() if use_cuda else "cpu"
        self.torch_dtype = torch.bfloat16

        # Will be initialized in _load_model
        self.pipeline = None

        self._load_model()

    def _select_best_device(self) -> str:
        """
        Select the best available GPU device based on free memory.

        Returns:
            Device string (cuda:X or cpu)
        """
        if not torch.cuda.is_available():
            return "cpu"

        try:
            # Get free memory for each GPU
            gpu_memory = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
            best_gpu_id = gpu_memory.index(max(gpu_memory))

            memory_gb = max(gpu_memory) / (1024**3)
            print(f"Selected GPU cuda:{best_gpu_id} with {memory_gb:.1f}GB free memory")
            return f"cuda:{best_gpu_id}"

        except Exception as e:
            print(f"GPU selection failed ({e}), falling back to cuda:0")
            return "cuda:0"
    
    def _load_model(self) -> None:
        """
        Load the diffusion model and apply optimizations.

        Raises:
            FileNotFoundError: If the model directory doesn't exist
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model directory not found: {self.model_path}. "
                "Please run 'python scripts/download_models.py' first."
            )

        print(f"Loading Qwen image generation model from {self.model_path}...")

        # Load the pre-trained pipeline
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            local_files_only=True  # Don't try to download
        )

        if self.use_memory_efficient:
            self._enable_memory_optimizations()
        else:
            # Move to GPU and enable basic optimizations
            self.pipeline = self.pipeline.to(self.device)
            self.pipeline.enable_attention_slicing(2)

        print("Model loaded successfully")

    def _enable_memory_optimizations(self) -> None:
        """Apply memory-saving optimizations for lower-end GPUs."""
        print("Applying memory-efficient optimizations...")

        # Offload model components to CPU when not in use
        self.pipeline.enable_model_cpu_offload()

        # Reduce memory usage during generation
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_attention_slicing(2)

        # Enable tiling for very large images (if supported)
        if hasattr(self.pipeline, 'enable_vae_tiling'):
            self.pipeline.enable_vae_tiling()
    
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
        """
        Generate an image from a text prompt.

        Args:
            prompt: The main text prompt describing the image
            negative_prompt: Text describing what to avoid in the image
            width: Image width in pixels
            height: Image height in pixels
            num_inference_steps: Number of denoising steps (higher = better quality)
            true_cfg_scale: Classifier-free guidance scale (higher = more faithful to prompt)
            seed: Random seed for reproducible generation

        Returns:
            Generated PIL Image
        """
        # Set up random generator for reproducible results
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Determine device type for mixed precision
        device_type = "cuda" if "cuda" in self.device else "cpu"

        # Generate with automatic mixed precision for speed and memory efficiency
        with torch.autocast(device_type=device_type, dtype=self.torch_dtype):
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale,
                generator=generator
            )

        return result.images[0]
    
    def generate_if_new(
        self,
        prompt: str,
        output_dir: str,
        filename_prefix: str = "",
        **kwargs
    ) -> Optional[str]:
        """
        Generate an image only if it doesn't already exist.

        This method creates a hash of the prompt to create a unique filename,
        checks if the image already exists, and only generates it if needed.

        Args:
            prompt: Text prompt for image generation
            output_dir: Directory to save the generated image
            filename_prefix: Optional prefix for the filename
            **kwargs: Additional arguments passed to generate()

        Returns:
            Path to the generated image, or None if it already existed
        """
        # Create a unique filename based on the prompt content
        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()

        # Build filename with optional prefix
        if filename_prefix:
            filename = f"{filename_prefix}_{prompt_hash}.jpg"
        else:
            filename = f"{prompt_hash}.jpg"

        output_path = os.path.join(output_dir, filename)

        # Skip if image already exists
        if os.path.exists(output_path):
            truncated_prompt = prompt[:30] + "..." if len(prompt) > 30 else prompt
            print(f"Skipping existing image: {truncated_prompt}")
            return None

        # Generate and save the new image
        truncated_prompt = prompt[:30] + "..." if len(prompt) > 30 else prompt
        print(f"Generating image: {truncated_prompt}")

        image = self.generate(prompt, **kwargs)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        self.save_image(image, output_path)

        return output_path
    
    def save_image(self, image: Image.Image, output_path: str) -> None:
        """
        Save an image to disk.

        Args:
            image: PIL Image to save
            output_path: File path where to save the image
        """
        image.save(output_path)
        print(f"Image saved to: {output_path}")


def main() -> None:
    """Demo function showing basic image generation usage."""
    # Initialize the generator with default settings
    generator = QwenImageGenerator()

    # Example prompt in Korean (a scene with a grandmother and grandson)
    prompt = "지난주 이마트 앞에서 카카오택시로 택시를 잡으려는 손자와 할머니를 봤습니다."
    output_dir = "./outputs"

    # Generate image with a demo prefix
    output_path = generator.generate_if_new(
        prompt,
        output_dir,
        filename_prefix="demo"
    )

    if output_path:
        print(f"Generation complete: {output_path}")
    else:
        print("Image already existed, skipped generation")


if __name__ == "__main__":
    main()
