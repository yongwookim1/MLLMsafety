# Scripts Usage Guide

## 1. Download Models (First Time Only)

Download models from Hugging Face to local storage:

```bash
python scripts/download_models.py
```

**Note**: This requires Hugging Face access. After downloading, copy the `models_cache/` directory to your server.

## 2. Generate Images

Generate images from KOBBQ contexts using Qwen-Image:

```bash
python scripts/generate_images.py
```

This will:
- Load unique contexts from KOBBQ dataset
- Generate images using Qwen-Image
- Save images to `outputs/kobbq_images/`

## 3. Run Evaluation

Run the complete evaluation pipeline:

```bash
python scripts/run_evaluation.py
```

This will:
- Create image-context mapping
- Run text-only vs multimodal comparison
- Evaluate results
- Find mismatch cases

## 4. Run Full Pipeline

Run all steps in sequence:

```bash
python scripts/run_full_pipeline.py
```

This executes:
1. Model availability check
2. Image generation
3. Mapping creation
4. Comparison
5. Evaluation
6. Mismatch analysis

## Running from Different Directories

All scripts automatically set the working directory to the project root, so you can run them from anywhere:

```bash
cd /path/to/MLLM_safety_qwen
python scripts/generate_images.py

# Or from scripts directory
cd scripts
python generate_images.py
```

