import os
import torch
from diffusers import DiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Optional
import yaml


class ImageGenerator:
    """
    Unified interface for generating images using different diffusion models.

    Supports both Qwen and Kimchi (Korean-specific) image generation models,
    with automatic device selection and memory management.
    """

    def __init__(self, config_path: str = "configs/config.yaml", device: Optional[str] = None):
        """
        Initialize the image generator with configuration.

        Args:
            config_path: Path to the YAML configuration file
            device: Override device selection (cuda:X or cpu)
        """
        # Load configuration
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Extract configuration sections
        self.model_config = self.config["models"]["image_generator"]
        self.gen_config = self.config["image_generation"]
        self.device_config = self.config["device"]

        # Determine compute device
        if device:
            self.device = device
        else:
            use_cuda = self.device_config.get("use_cuda", True) and torch.cuda.is_available()
            self.device = self.device_config.get("cuda_device", "cuda:0") if use_cuda else "cpu"

        # Set data type for the model
        self.torch_dtype = getattr(torch, self.model_config["torch_dtype"])
        self.model_type = self.model_config.get("type", "qwen-image")

        # Pipeline will be initialized by _load_model
        self.pipeline = None

        # Memory management settings
        self._cache_clear_interval = max(1, self.gen_config.get("cache_clear_interval", 32))
        self._steps_since_cache_clear = 0

        # Load the appropriate model
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Load the appropriate model based on configuration.

        Routes to either Qwen or Kimchi model loading.
        """
        if self.model_type == "kimchi":
            self._load_kimchi_model()
        else:
            self._load_qwen_model()
    
    def _load_qwen_model(self) -> None:
        """
        Load the Qwen image generation model with optimizations.

        Raises:
            FileNotFoundError: If model directory doesn't exist
        """
        model_path = self.model_config["local_path"]
        print(f"Loading Qwen image generation model from {model_path}...")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Qwen model not found at {model_path}. "
                "Please run 'python scripts/download_models.py' first."
            )

        # Load the pre-trained diffusion pipeline
        self.pipeline = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=self.torch_dtype,
            local_files_only=True
        )

        # Apply memory optimizations if requested
        if self.model_config.get("use_memory_efficient", False):
            self._apply_memory_optimizations()
        else:
            # Standard setup for higher-end GPUs
            self.pipeline = self.pipeline.to(self.device)
            self.pipeline.enable_attention_slicing(2)

        print("Qwen model loaded successfully")

    def _apply_memory_optimizations(self) -> None:
        """Apply aggressive memory-saving optimizations for lower-end GPUs."""
        print("Applying memory-efficient optimizations...")

        # Clear any leftover GPU memory first
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Enable CPU offloading for model components
        self.pipeline.enable_model_cpu_offload()

        # Reduce memory usage during generation
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_attention_slicing(8)  # More aggressive than default

        # Move VAE to CPU to save GPU memory
        if hasattr(self.pipeline, 'vae'):
            self.pipeline.vae.to('cpu')
            print("VAE moved to CPU for memory efficiency")

        # Enable tiling for large images if supported
        if hasattr(self.pipeline, 'enable_vae_tiling'):
            self.pipeline.enable_vae_tiling()

        # Note: xFormers disabled due to compatibility issues
        print("Using standard attention mechanism (xFormers disabled for compatibility)")

        # Reserve some GPU memory for system processes
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.9)
    
    def _load_kimchi_model(self) -> None:
        """
        Load the Kimchi model (Stable Diffusion fine-tuned for Korean with CLIP and LoRA).

        Raises:
            FileNotFoundError: If any required model components are missing
        """
        print("Loading Kimchi model (Korean-optimized Stable Diffusion)...")

        # Extract model paths from configuration
        pretrained_path = self.model_config.get("pretrained_model_path", "./models_cache/stable-diffusion-v1-5")
        lora_path = self.model_config.get("lora_path")
        clip_path = self.model_config.get("clip_model_path", "./models_cache/clip-vit-large-patch14-ko")

        # Validate that all required models exist
        self._validate_kimchi_model_paths(pretrained_path, clip_path, lora_path)

        # Load base Stable Diffusion model
        print(f"Loading base Stable Diffusion model from {pretrained_path}...")
        self.pipeline = DiffusionPipeline.from_pretrained(
            pretrained_path,
            torch_dtype=self.torch_dtype,
            local_files_only=True
        )

        # Replace with Korean CLIP model
        print(f"Loading Korean CLIP text encoder from {clip_path}...")
        korean_encoder = CLIPTextModel.from_pretrained(clip_path, local_files_only=True)
        korean_tokenizer = CLIPTokenizer.from_pretrained(clip_path, local_files_only=True)

        # Update pipeline with Korean components
        self.pipeline.text_encoder = korean_encoder.to(self.device)
        self.pipeline.tokenizer = korean_tokenizer

        # Load LoRA weights if specified
        if lora_path:
            print(f"Loading LoRA fine-tuning weights from {lora_path}...")
            self.pipeline.load_lora_weights(lora_path)

        # Move to device and enable optimizations
        self.pipeline = self.pipeline.to(self.device)
        self.pipeline.enable_attention_slicing(2)

        print("Kimchi model loaded successfully")

    def _validate_kimchi_model_paths(self, pretrained_path: str, clip_path: str, lora_path: Optional[str]) -> None:
        """
        Validate that all Kimchi model components exist.

        Args:
            pretrained_path: Path to base Stable Diffusion model
            clip_path: Path to Korean CLIP model
            lora_path: Path to LoRA weights (optional)

        Raises:
            FileNotFoundError: If any required path doesn't exist
        """
        if not os.path.exists(pretrained_path) or not os.path.isdir(pretrained_path):
            raise FileNotFoundError(f"Stable Diffusion model not found at {pretrained_path}")

        if not os.path.exists(clip_path) or not os.path.isdir(clip_path):
            raise FileNotFoundError(f"Korean CLIP model not found at {clip_path}")

        if lora_path and not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA weights not found at {lora_path}")
    
    def generate(self, prompt: str, negative_prompt: str = " ", seed: Optional[int] = None):
        """
        Generate an image from a text prompt using the configured model.

        Args:
            prompt: Text description of the image to generate
            negative_prompt: Text describing what to avoid in the image
            seed: Random seed for reproducible generation

        Returns:
            Generated PIL Image

        Raises:
            ValueError: If the pipeline fails to generate images
            Exception: Any error during generation (after cleanup)
        """
        # Calculate dimensions based on configured aspect ratio
        width, height = self._calculate_image_dimensions()

        # Set up random generator for reproducible results
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        device_type = "cuda" if "cuda" in self.device else "cpu"

        try:
            if self.model_type == "kimchi":
                image = self._generate_with_kimchi_model(prompt, device_type)
            else:
                image = self._generate_with_qwen_model(
                    prompt, negative_prompt, width, height, generator, device_type
                )

            return image

        except Exception as e:
            print(f"Image generation failed: {e}")
            self._force_cache_clear()
            raise
        finally:
            self._schedule_cache_clear()

    def _calculate_image_dimensions(self) -> tuple[int, int]:
        """
        Calculate image dimensions based on configured aspect ratio.

        Returns:
            Tuple of (width, height) in pixels
        """
        base_width = self.gen_config.get("width", 512)
        base_height = self.gen_config.get("height", 512)

        # Common aspect ratio mappings
        aspect_ratios = {
            "1:1": (base_width, base_height),
            "16:9": (int(base_width * 1.78), base_height),  # Landscape
            "9:16": (base_width, int(base_height * 1.78)),  # Portrait
            "4:3": (int(base_width * 1.33), base_height),
            "3:4": (base_width, int(base_height * 1.33)),
            "3:2": (int(base_width * 1.5), base_height),
            "2:3": (base_width, int(base_height * 1.5)),
        }

        aspect_ratio = self.gen_config.get("aspect_ratio", "1:1")
        return aspect_ratios.get(aspect_ratio, (base_width, base_height))

    def _generate_with_kimchi_model(self, prompt: str, device_type: str):
        """Generate image using the Kimchi (Korean) model."""
        with torch.autocast(device_type=device_type):
            result = self.pipeline(
                prompt,
                num_inference_steps=self.gen_config["num_inference_steps"]
            )

        if not hasattr(result, 'images') or not result.images:
            raise ValueError("Kimchi model pipeline returned no images")

        return result.images[0]

    def _generate_with_qwen_model(self, prompt: str, negative_prompt: str, width: int, height: int,
                                  generator, device_type: str):
        """Generate image using the Qwen model."""
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

        if not hasattr(result, 'images') or len(result.images) == 0:
            raise ValueError(f"Qwen model pipeline returned no images (result type: {type(result)})")

        return result.images[0]

    def _schedule_cache_clear(self) -> None:
        """
        Periodically clear CUDA cache to prevent memory fragmentation.

        Clears cache every N generations as configured to balance memory usage
        and performance.
        """
        if not torch.cuda.is_available():
            return

        self._steps_since_cache_clear += 1
        if self._steps_since_cache_clear >= self._cache_clear_interval:
            torch.cuda.empty_cache()
            self._steps_since_cache_clear = 0

    def _force_cache_clear(self) -> None:
        """Force immediate clearing of CUDA cache."""
        if not torch.cuda.is_available():
            return

        torch.cuda.empty_cache()
        self._steps_since_cache_clear = 0

