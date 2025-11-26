import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import yaml
from huggingface_hub import snapshot_download

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

DEFAULT_LORA_SOURCE = project_root.parent / "MLLM_safety_kimchi" / "outputs" / "korean-4-datasets" / "checkpoint-37450" / "pytorch_lora_weights.safetensors"


def download_repo(repo_id: str, local_path: str, description: str) -> bool:
    abs_path = os.path.abspath(local_path)
    print(f"{description}")
    print(f"Repo: {repo_id}")
    print(f"Path: {abs_path}")
    
    os.makedirs(abs_path, exist_ok=True)
    
    if any(os.scandir(abs_path)):
        print("Already exists")
        print()
        return True
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=abs_path,
            local_dir_use_symlinks=False
        )
        print("Done")
        print()
        return True
    except Exception as e:
        print(f"Failed: {e}")
        print()
        return False


def copy_lora_weights(destination: str, source: Optional[str]) -> bool:
    dest_path = os.path.abspath(destination)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    if os.path.exists(dest_path):
        print(f"LoRA weights already at {dest_path}")
        return True
    
    candidate_sources = [
        source,
        os.environ.get("KIMCHI_LORA_SOURCE"),
        str(DEFAULT_LORA_SOURCE),
    ]
    
    for candidate in candidate_sources:
        if not candidate:
            continue
        candidate_path = os.path.abspath(candidate)
        if os.path.exists(candidate_path):
            try:
                shutil.copy2(candidate_path, dest_path)
                print(f"Copied LoRA from {candidate_path}")
                return True
            except Exception as exc:
                print(f"Copy failed: {exc}")
                return False
    
    print("LoRA weights not found")
    print(f"Place manually at: {dest_path}")
    print("(or set KIMCHI_LORA_SOURCE env var)")
    return False


def download_qwen_image(model_cfg: dict) -> bool:
    return download_repo(
        repo_id=model_cfg["name"],
        local_path=model_cfg["local_path"],
        description="Image Generator (Qwen-Image)"
    )


def download_kimchi_assets(model_cfg: dict) -> bool:
    pretrained_repo = model_cfg.get("pretrained_model_repo", "runwayml/stable-diffusion-v1-5")
    clip_repo = model_cfg.get("clip_model_repo", "Bingsu/clip-vit-large-patch14-ko")
    
    print("Downloading Kimchi image generator assets...")
    success = True
    success &= download_repo(pretrained_repo, model_cfg["pretrained_model_path"], "Stable Diffusion v1.5 base")
    success &= download_repo(clip_repo, model_cfg["clip_model_path"], "Korean CLIP text encoder")
    success &= copy_lora_weights(model_cfg["lora_path"], model_cfg.get("lora_source_path"))
    return success


def download_evaluator(model_cfg: dict) -> bool:
    return download_repo(
        repo_id=model_cfg["name"],
        local_path=model_cfg["local_path"],
        description="Evaluator (Qwen2.5-VL)"
    )


def download_judge(model_cfg: dict) -> bool:
    return download_repo(
        repo_id=model_cfg["name"],
        local_path=model_cfg["local_path"],
        description="Judge (Qwen2.5-7B-Instruct)"
    )


def main():
    config_path = "configs/config.yaml"
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    models_config = config["models"]
    image_cfg = models_config["image_generator"]
    evaluator_cfg = models_config["evaluator"]
    judge_cfg = models_config.get("judge")
    
    print("Model Download Script")
    print()
    
    overall_success = True
    
    model_type = image_cfg.get("type", "qwen-image").lower()
    if model_type == "kimchi":
        overall_success &= download_kimchi_assets(image_cfg)
    else:
        overall_success &= download_qwen_image(image_cfg)
    
    overall_success &= download_evaluator(evaluator_cfg)

    if judge_cfg:
        overall_success &= download_judge(judge_cfg)
    
    if overall_success:
        print("All downloads completed")
    else:
        print("Some downloads failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

