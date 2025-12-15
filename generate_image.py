import os
import torch
import torch.distributed as dist
import hashlib
from diffusers import DiffusionPipeline
from PIL import Image
from typing import Optional, List, Union

try:
    from diffusers import ZImagePipeline
    ZIMAGE_AVAILABLE = True
except ImportError:
    ZIMAGE_AVAILABLE = False


def setup_distributed():
    """Initialize distributed environment if not already initialized."""
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size
    return 0, 1


def is_main_process():
    """Check if current process is the main process."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


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


class ZImageGenerator:
    """Image generator using Z-Image-Turbo model with batch and distributed support."""

    def __init__(
        self,
        model_path: str = "./models_cache/z-image-turbo",
        use_cuda: bool = True,
        use_memory_efficient: bool = False,
        use_distributed: bool = False
    ):
        if not ZIMAGE_AVAILABLE:
            raise ImportError(
                "ZImagePipeline not available. Install diffusers from source:\n"
                "pip install git+https://github.com/huggingface/diffusers"
            )
        
        self.model_path = os.path.abspath(model_path)
        self.use_memory_efficient = use_memory_efficient
        self.use_distributed = use_distributed
        self.torch_dtype = torch.bfloat16
        self.pipeline = None
        
        if use_distributed:
            self.rank, self.world_size = setup_distributed()
            self.device = f"cuda:{self.rank % torch.cuda.device_count()}"
        else:
            self.rank, self.world_size = 0, 1
            self.device = self._select_best_device() if use_cuda else "cpu"
        
        self._load_model()

    def _select_best_device(self) -> str:
        if not torch.cuda.is_available():
            return "cpu"
        try:
            gpu_memory = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
            best_gpu_id = gpu_memory.index(max(gpu_memory))
            if is_main_process():
                print(f"Selected GPU cuda:{best_gpu_id} with {max(gpu_memory) / (1024**3):.1f}GB free memory")
            return f"cuda:{best_gpu_id}"
        except Exception:
            return "cuda:0"

    def _load_model(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model directory not found: {self.model_path}. "
                "Please download the model first using:\n"
                "  huggingface-cli download Tongyi-MAI/Z-Image-Turbo --local-dir ./models_cache/z-image-turbo"
            )

        if is_main_process():
            print(f"Loading Z-Image-Turbo model from {self.model_path}...")

        self.pipeline = ZImagePipeline.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=False,
            local_files_only=True
        )

        if self.use_memory_efficient:
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline.to(self.device)

        if is_main_process():
            print("Z-Image-Turbo model loaded successfully")

    def generate(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 9,
        guidance_scale: float = 0.0,
        seed: Optional[int] = None,
        **kwargs
    ) -> Union[Image.Image, List[Image.Image]]:
        """Generate image(s) from prompt(s). Supports both single and batch."""
        generator = None
        if seed is not None:
            generator = torch.Generator(self.device).manual_seed(seed)

        result = self.pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        if isinstance(prompt, str):
            return result.images[0]
        return result.images

    def generate_batch(
        self,
        prompts: List[str],
        batch_size: int = 8,
        **kwargs
    ) -> List[Image.Image]:
        """Generate images in batches."""
        all_images = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            images = self.generate(batch_prompts, **kwargs)
            if isinstance(images, Image.Image):
                images = [images]
            all_images.extend(images)
        return all_images

    def generate_distributed(
        self,
        prompts: List[str],
        batch_size: int = 8,
        **kwargs
    ) -> List[Image.Image]:
        """Generate images with distributed processing across multiple GPUs."""
        # Split prompts across ranks
        prompts_per_rank = len(prompts) // self.world_size
        start_idx = self.rank * prompts_per_rank
        end_idx = start_idx + prompts_per_rank if self.rank < self.world_size - 1 else len(prompts)
        local_prompts = prompts[start_idx:end_idx]
        
        if is_main_process():
            print(f"Rank {self.rank}: processing {len(local_prompts)} prompts")
        
        local_images = self.generate_batch(local_prompts, batch_size=batch_size, **kwargs)
        return local_images

    def generate_if_new(
        self,
        prompt: str,
        output_dir: str,
        filename_prefix: str = "",
        **kwargs
    ) -> Optional[str]:
        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        filename = f"{filename_prefix}_{prompt_hash}.jpg" if filename_prefix else f"{prompt_hash}.jpg"
        output_path = os.path.join(output_dir, filename)

        if os.path.exists(output_path):
            if is_main_process():
                print(f"Skipping existing image: {prompt[:30]}...")
            return None

        if is_main_process():
            print(f"Generating image: {prompt[:30]}...")
        image = self.generate(prompt, **kwargs)
        os.makedirs(output_dir, exist_ok=True)
        self.save_image(image, output_path)
        return output_path

    def generate_batch_if_new(
        self,
        prompts: List[str],
        output_dir: str,
        filename_prefix: str = "",
        batch_size: int = 8,
        **kwargs
    ) -> List[str]:
        """Generate multiple images, skipping existing ones. Supports distributed."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter out existing images
        prompts_to_generate = []
        output_paths = []
        for prompt in prompts:
            prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
            filename = f"{filename_prefix}_{prompt_hash}.jpg" if filename_prefix else f"{prompt_hash}.jpg"
            output_path = os.path.join(output_dir, filename)
            if not os.path.exists(output_path):
                prompts_to_generate.append(prompt)
                output_paths.append(output_path)
        
        if not prompts_to_generate:
            if is_main_process():
                print("All images already exist, skipping generation")
            return []
        
        if is_main_process():
            print(f"Generating {len(prompts_to_generate)} new images...")
        
        if self.use_distributed and self.world_size > 1:
            local_prompts_idx = list(range(self.rank, len(prompts_to_generate), self.world_size))
            local_prompts = [prompts_to_generate[i] for i in local_prompts_idx]
            local_paths = [output_paths[i] for i in local_prompts_idx]
            images = self.generate_batch(local_prompts, batch_size=batch_size, **kwargs)
        else:
            local_paths = output_paths
            images = self.generate_batch(prompts_to_generate, batch_size=batch_size, **kwargs)
        
        for img, path in zip(images, local_paths):
            img.save(path)
        
        if self.use_distributed:
            dist.barrier()
        
        return output_paths

    def save_image(self, image: Image.Image, output_path: str) -> None:
        image.save(output_path)
        if is_main_process():
            print(f"Image saved to: {output_path}")


def get_image_generator(generator_type: str = "qwen-image", **kwargs):
    """Factory function to create image generator based on type."""
    generators = {
        "qwen-image": QwenImageGenerator,
        "z-image-turbo": ZImageGenerator,
    }
    if generator_type not in generators:
        raise ValueError(f"Unknown generator type: {generator_type}. Available: {list(generators.keys())}")
    return generators[generator_type](**kwargs)


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
