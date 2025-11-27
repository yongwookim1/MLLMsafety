import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, GenerationConfig
from typing import Optional, Any, List, Dict
from PIL import Image
import yaml


class Evaluator:
    def __init__(self, config_path: str = "configs/config.yaml", device_map: Optional[Any] = None):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config["models"]["evaluator"]
        self.eval_config = self.config["evaluation"]
        self.device_config = self.config["device"]
        
        self.device = self.device_config["cuda_device"] if self.device_config["use_cuda"] and torch.cuda.is_available() else "cpu"
        self.torch_dtype = getattr(torch, self.model_config["torch_dtype"])
        
        self.model = None
        self.processor = None
        self._load_model(device_map)
        self.generation_config = GenerationConfig(
            max_new_tokens=self.eval_config["max_new_tokens"],
            do_sample=False,
        )
    
    def _load_model(self, device_map_override: Optional[Any] = None):
        print(f"Loading Qwen2.5-VL model from {self.model_config['local_path']}...")
        
        if not os.path.exists(self.model_config["local_path"]):
            raise FileNotFoundError(
                f"Model not found at {self.model_config['local_path']}. "
                f"Run scripts/download_models.py first."
            )
        
        if device_map_override is not None:
            device_map = device_map_override
        else:
            device_map = self.model_config.get("device_map", "auto")
            if device_map == "auto" and self.device_config["use_cuda"]:
                device_map = {"": self.device_config["cuda_device"]}
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_config["local_path"],
            torch_dtype=self.torch_dtype,
            device_map=device_map,
            local_files_only=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_config["local_path"],
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

        print("Model loaded")
    
    def run_inference(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        use_image: bool = True
    ) -> str:
        return self.run_batch([
            {"prompt": prompt, "image": image, "use_image": use_image}
        ])[0]

    def run_batch(self, requests: List[Dict[str, Any]]) -> List[str]:
        if not requests:
            return []

        responses: List[Optional[str]] = [None] * len(requests)

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
            for idx, resp in zip(text_only_indices, outputs):
                responses[idx] = resp

        image_indices = [
            idx for idx, req in enumerate(requests)
            if req.get("use_image") and req.get("image") is not None
        ]
        if image_indices:
            texts = [
                self._build_chat_text(
                    requests[idx]["prompt"],
                    requests[idx]["image"],
                    True
                )
                for idx in image_indices
            ]
            images = [requests[idx]["image"] for idx in image_indices]
            outputs = self._generate(texts, images=images)
            for idx, resp in zip(image_indices, outputs):
                responses[idx] = resp

        for idx, resp in enumerate(responses):
            if resp is None:
                req = requests[idx]
                use_image = req.get("use_image", False) and req.get("image") is not None
                text = self._build_chat_text(req["prompt"], req.get("image"), use_image)
                single_output = self._generate(
                    [text],
                    images=[req.get("image")] if use_image else None
                )
                responses[idx] = single_output[0] if single_output else ""

        return responses

    def _build_chat_text(
        self,
        prompt: str,
        image: Optional[Image.Image],
        use_image: bool
    ) -> str:
        if use_image and image is not None:
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
        if not texts:
            return []

        if images is not None:
            inputs = self.processor(
                text=texts,
                images=images,
                padding=True,
                return_tensors="pt"
            )
        else:
            inputs = self.processor(
                text=texts,
                padding=True,
                return_tensors="pt"
            )

        inputs = inputs.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
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
