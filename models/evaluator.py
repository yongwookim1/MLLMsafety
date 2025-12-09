import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, GenerationConfig, AutoModel
from typing import Optional, Any, List, Dict
from PIL import Image
import yaml


class Evaluator:
    """
    Multimodal evaluator using Qwen2.5-VL or Qwen3-VL model for text and image analysis.

    Handles both text-only and text+image evaluation tasks with proper
    device management and memory optimization.
    """

    def __init__(self, config_path: str = "configs/config.yaml", device_map: Optional[Any] = None):
        """
        Initialize the multimodal evaluator.

        Args:
            config_path: Path to YAML configuration file
            device_map: Optional device mapping for model distribution
        """
        # Load and parse configuration
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Extract configuration sections
        self.model_config = self.config["models"]["evaluator"]
        self.eval_config = self.config["evaluation"]
        self.device_config = self.config["device"]

        # Determine compute device
        self.device = self.device_config["cuda_device"] if self.device_config["use_cuda"] and torch.cuda.is_available() else "cpu"

        # Override device if device_map specifies one
        if device_map is not None and isinstance(device_map, dict) and "" in device_map:
            self.device = device_map[""]

        # Set model data type
        self.torch_dtype = getattr(torch, self.model_config["torch_dtype"])

        # Model components (initialized by _load_model)
        self.model = None
        self.processor = None

        # Load the model
        self._load_model(device_map)

        # Configure generation parameters
        self.generation_config = GenerationConfig(
            max_new_tokens=self.eval_config["max_new_tokens"],
            do_sample=False,  # Deterministic generation for evaluation
        )
    
    def _load_model(self, device_map_override: Optional[Any] = None) -> None:
        """
        Load the Qwen2.5-VL or Qwen3-VL evaluation model and processor.

        Args:
            device_map_override: Override the default device mapping

        Raises:
            FileNotFoundError: If model directory doesn't exist
        """
        model_path = self.model_config["local_path"]
        model_name = self.model_config.get("name", "").lower()
        print(f"Loading evaluation model from {model_path}...")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please run 'python scripts/download_models.py' first."
            )

        # Determine device mapping strategy
        if device_map_override is not None:
            device_map = device_map_override
        else:
            device_map = self.model_config.get("device_map", "auto")
            # Auto device mapping for CUDA if enabled
            if device_map == "auto" and self.device_config.get("use_cuda", True):
                device_map = {"": self.device_config["cuda_device"]}
        
        # Check for Qwen3
        is_qwen3 = "qwen3" in model_name or "qwen3" in model_path.lower()

        if is_qwen3:
            print("Detected Qwen3-VL model. Loading with AutoModel...")
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=self.torch_dtype,
                device_map=device_map,
                local_files_only=True,
                trust_remote_code=True
            )
        else:
            print("Loading with Qwen2_5_VLForConditionalGeneration...")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=self.torch_dtype,
                device_map=device_map,
                local_files_only=True
            )

        # Load the processor for text and image handling
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            padding_side="left",  # Better for generation tasks
            trust_remote_code=True if is_qwen3 else False
        )

        # Configure tokenizer for generation
        self._configure_tokenizer()

        print("Model loaded successfully")

    def _configure_tokenizer(self) -> None:
        """Configure tokenizer settings for proper generation."""
        tokenizer = getattr(self.processor, 'tokenizer', None)
        if tokenizer is not None:
            # Ensure left padding for generation
            tokenizer.padding_side = 'left'
            # Set pad token if not already set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        # Configure processor-level padding
        if hasattr(self.processor, 'padding_side'):
            self.processor.padding_side = 'left'
        if getattr(self.processor, 'pad_token', None) is None and hasattr(self.processor, 'eos_token'):
            self.processor.pad_token = self.processor.eos_token
    
    def _resize_image_if_needed(self, image: Image.Image, max_size: int = 1024) -> Image.Image:
        """
        Resize image if it exceeds maximum dimensions to prevent out-of-memory errors.

        Args:
            image: Input PIL Image
            max_size: Maximum allowed dimension (width or height)

        Returns:
            Resized image or original if already within limits
        """
        width, height = image.size
        max_dimension = max(width, height)

        if max_dimension > max_size:
            # Calculate scaling ratio to fit within max_size
            scale_ratio = max_size / max_dimension
            new_size = (int(width * scale_ratio), int(height * scale_ratio))

            # Use high-quality Lanczos resampling
            return image.resize(new_size, Image.Resampling.LANCZOS)

        return image

    def run_inference(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        use_image: bool = True
    ) -> str:
        """
        Run inference on a single prompt, optionally with an image.

        Args:
            prompt: Text prompt for the model
            image: Optional image to include in the context
            use_image: Whether to include the image in processing

        Returns:
            Model's text response
        """
        return self.run_batch([
            {"prompt": prompt, "image": image, "use_image": use_image}
        ])[0]

    def run_batch(self, requests: List[Dict[str, Any]]) -> List[str]:
        """
        Process multiple inference requests in batch for efficiency.

        Args:
            requests: List of request dictionaries with 'prompt', 'image', and 'use_image' keys

        Returns:
            List of model responses corresponding to each request
        """
        if not requests:
            return []

        # Pre-process all images to prevent OOM errors
        self._preprocess_images_in_requests(requests)

        # Initialize response list
        responses: List[Optional[str]] = [None] * len(requests)

        # Process text-only requests
        self._process_text_only_requests(requests, responses)

        # Process image+text requests
        self._process_multimodal_requests(requests, responses)

        # Handle any remaining unprocessed requests
        self._process_remaining_requests(requests, responses)

        return responses

    def _preprocess_images_in_requests(self, requests: List[Dict[str, Any]]) -> None:
        """Resize images in requests to prevent memory issues."""
        for request in requests:
            if request.get("use_image") and request.get("image") is not None:
                request["image"] = self._resize_image_if_needed(request["image"])

    def _process_text_only_requests(self, requests: List[Dict[str, Any]], responses: List[Optional[str]]) -> None:
        """Process requests that only contain text."""
        text_only_indices = [
            idx for idx, req in enumerate(requests)
            if not (req.get("use_image") and req.get("image") is not None)
        ]

        if text_only_indices:
            texts = [
                self._build_chat_text(requests[idx]["prompt"], None, False)
                for idx in text_only_indices
            ]
            outputs = self._generate(texts, images=None)
            for idx, response in zip(text_only_indices, outputs):
                responses[idx] = response

    def _process_multimodal_requests(self, requests: List[Dict[str, Any]], responses: List[Optional[str]]) -> None:
        """Process requests that include both text and images."""
        image_indices = [
            idx for idx, req in enumerate(requests)
            if req.get("use_image") and req.get("image") is not None
        ]

        if image_indices:
            texts = [
                self._build_chat_text(requests[idx]["prompt"], requests[idx]["image"], True)
                for idx in image_indices
            ]
            images = [requests[idx]["image"] for idx in image_indices]
            outputs = self._generate(texts, images=images)
            for idx, response in zip(image_indices, outputs):
                responses[idx] = response

    def _process_remaining_requests(self, requests: List[Dict[str, Any]], responses: List[Optional[str]]) -> None:
        """Process any requests that weren't handled in the batch operations."""
        for idx, response in enumerate(responses):
            if response is None:
                request = requests[idx]
                use_image = request.get("use_image", False) and request.get("image") is not None
                text = self._build_chat_text(request["prompt"], request.get("image"), use_image)
                single_output = self._generate(
                    [text],
                    images=[request.get("image")] if use_image else None
                )
                responses[idx] = single_output[0] if single_output else ""

    def _build_chat_text(
        self,
        prompt: str,
        image: Optional[Image.Image],
        use_image: bool
    ) -> str:
        """
        Build a chat-formatted message for the model.

        Args:
            prompt: Text prompt from user
            image: Optional image to include
            use_image: Whether to include the image in the message

        Returns:
            Formatted chat template string
        """
        if use_image and image is not None:
            # Multimodal message with image and text
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
            # Text-only message
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _generate(
        self,
        texts: List[str],
        images: Optional[List[Image.Image]]
    ) -> List[str]:
        """
        Generate responses from the model for given texts and optional images.

        Args:
            texts: List of formatted text prompts
            images: Optional list of images corresponding to texts

        Returns:
            List of generated text responses
        """
        if not texts:
            return []

        # Prepare inputs for the model
        inputs = self._prepare_model_inputs(texts, images)

        # Move inputs to the appropriate device
        inputs = inputs.to(self.device)

        # Generate responses without gradient computation
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
            )

        # Remove input tokens from generated sequences
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode the generated tokens to text
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

    def _prepare_model_inputs(self, texts: List[str], images: Optional[List[Image.Image]]):
        """
        Prepare tokenized inputs for the model.

        Args:
            texts: List of text prompts
            images: Optional list of images

        Returns:
            Tokenized inputs ready for model consumption
        """
        if images is not None:
            # Multimodal inputs
            return self.processor(
                text=texts,
                images=images,
                padding=True,
                return_tensors="pt"
            )
        else:
            # Text-only inputs
            return self.processor(
                text=texts,
                padding=True,
                return_tensors="pt"
            )
