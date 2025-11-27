# MLLM Safety Evaluation Pipeline

Multimodal Language Model safety evaluation pipeline using Qwen-Image for image generation and Qwen2.5-VL for evaluation. All models run locally without Hugging Face connection after initial download.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Models

Download models to local storage (requires internet connection):

```bash
python scripts/download_models.py
```

This downloads:
- **Qwen/Qwen-Image** → `models_cache/qwen-image/` (~10-20GB)
- **Qwen/Qwen2.5-VL-7B-Instruct** → `models_cache/qwen2.5-vl-7b-instruct/` (~15-20GB)

### 3. Download Dataset

Download KOBBQ dataset to local cache (requires internet connection):

```bash
python scripts/download_dataset.py
```

This downloads:
- **naver-ai/kobbq** → `data_cache/` (cached for offline use)

### 4. Run Pipeline

```bash
# Full pipeline
python scripts/run_full_pipeline.py

# Or step by step
python scripts/generate_images.py    # Generate images
python scripts/run_evaluation.py      # Run evaluation

# TTA Dataset Pipeline (Multimodal Augmentation)
python scripts/run_tta_pipeline.py    # Complete TTA pipeline
```

The evaluation step iterates over all five KoBBQ prompt templates defined in `evaluation/kobbq_prompts.tsv`, mirroring the official KoBBQ benchmark procedure.

## Project Structure

```
MLLMsafety/
├── configs/
│   └── config.yaml              # Configuration
├── models/
│   ├── image_generator.py      # Qwen-Image wrapper
│   └── evaluator.py            # Qwen2.5-VL wrapper
├── data/
│   ├── kobbq_loader.py         # KOBBQ dataset loader
│   └── hate_community_loader.py # Hate community dataset
├── evaluation/
│   └── evaluator.py            # Evaluation pipeline
├── scripts/
│   ├── download_models.py      # Download models
│   ├── download_dataset.py     # Download dataset
│   ├── generate_images.py      # Generate images
│   ├── run_evaluation.py       # Run evaluation
│   └── run_full_pipeline.py    # Full pipeline
├── models_cache/               # Local model storage
├── data_cache/                 # Dataset cache
└── outputs/                    # Results
```

## Configuration

Edit `configs/config.yaml` to adjust:
- Model paths and settings
- Image generation parameters
- Evaluation settings
- Batch sizes (`generation_batch_size`, `judge_batch_size`) to match your GPU VRAM
- Prompt templates / prompt IDs (KoBBQ instructions)
- Output directories

## Models

- **Image Generator**: Qwen-Image (`models_cache/qwen-image/`)
- **Evaluator**: Qwen2.5-VL-7B-Instruct (`models_cache/qwen2.5-vl-7b-instruct/`)
  - Options: 3B, 7B (default), 72B

## Dataset

- **KOBBQ**: Korean Bias Benchmark (`naver-ai/kobbq`)
- **Hate Community**: Local JSON dataset (`data/hate_community_dataset.json`)
- **TTA01/AssurAI**: Task Transfer Augmentation dataset for multimodal safety evaluation

### TTA Dataset Pipeline (Multimodal Augmentation)

For environments with HuggingFace firewall restrictions:

#### Manual Download (on machine with internet access)

```bash
# Download TTA dataset manually (includes ~230 image files)
python scripts/manual_download_tta.py

# Verify download (checks core files + image files)
python scripts/manual_download_tta.py --verify-only
```

#### Transfer to Server

```bash
# Transfer downloaded dataset to your server
scp -r data_cache/TTA01_AssurAI user@server:/path/to/MLLMsafety/data_cache/
```

#### Run TTA Pipeline

```bash
# Complete TTA pipeline: text-to-image augmentation + multimodal evaluation with comparison
python scripts/run_tta_pipeline.py

# Step by step
python scripts/prepare_tta_data.py    # Convert text samples to images
python scripts/run_tta_evaluation.py  # Run multimodal safety evaluation (text-only vs multimodal comparison)

# With sample limit (for testing)
python scripts/run_tta_pipeline.py --limit 10
```

## Deployment

### Transfer to Server

1. **Download models and dataset locally** (with internet):
   ```bash
   python scripts/download_models.py
   python scripts/download_dataset.py
   ```

2. **Transfer to server** (include `data_cache/` directory):
   ```bash
   # Using rsync (recommended)
   rsync -avz --progress MLLMsafety/ user@server:/path/to/destination/
   
   # Or using scp
   scp -r MLLMsafety user@server:/path/to/destination/
   ```

3. **On server**:
   ```bash
   cd /path/to/destination/MLLMsafety
   pip install -r requirements.txt
   python scripts/test_pipeline.py
   python scripts/run_full_pipeline.py
   ```

### Requirements

- **Disk Space**: ~50GB (models ~30-40GB)
- **GPU**: Recommended (8GB+ VRAM for image gen, 16GB+ for evaluator)
- **Network**: Required only for initial model download

## Output Files

- `outputs/hate_images/`: Generated images
- `outputs/evaluation_results/`: Evaluation summaries saved per prompt (e.g., `kobbq_comparison_evaluation_prompt_1.json`)
- `outputs/*.json`: Detailed comparison and mismatch logs per prompt plus the image-context mapping

## Troubleshooting

**Model not found**: Run `python scripts/download_models.py` first

**CUDA out of memory**: 
- Use smaller model (3B instead of 7B)
- Reduce batch size in config
- Enable memory-efficient mode

**Dataset not found**: 
- Run `python scripts/download_dataset.py` first
- Dataset uses `local_files_only=True` - requires pre-downloaded cache
- After download, `data_cache/` can be transferred to offline servers

## Notes

- Models and dataset use `local_files_only=True` - no Hugging Face connection needed after download
- All paths are automatically converted to absolute paths
- Models are stored in `models_cache/` (excluded from git)
- Dataset cache is stored in `data_cache/` (excluded from git)
- Both `models_cache/` and `data_cache/` should be transferred when deploying to offline servers
