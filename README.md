# Product Personalization with LoRA

A small, inspectable Python project for testing whether a product LoRA improves the visual fidelity of a real product over prompt-only and reference-conditioned generation. The first experiment targets a camping tent identified in captions as `zkttent`.

## Compatibility decision

The project deliberately uses **FLUX.2 Klein Base 4B** for training; it does not substitute another model. As of MFLUX 0.24.0, MFLUX supports `flux2-klein-base-4b` LoRA training on Apple Silicon, and its FLUX.2 implementation supports text-to-image, image-conditioned editing, quantization, and LoRA inference.

Important constraints:

- MFLUX/MLX local execution requires an Apple Silicon Mac. The `mflux` dependency is therefore installed only on macOS arm64; configuration and dataset tests remain usable elsewhere.
- The upstream 4B bf16 weights are about 13–15 GB. Training requires more unified memory for activations and optimizer state. The example enables 8-bit loading, low-RAM mode, gradient checkpointing, batch size 1, and 768 px training. MFLUX documents gradient checkpointing as making the heavier edit workflow fit on 32–36 GB Macs, at roughly a 30% speed cost. A 16 GB Mac may still run out of memory; lower the configured resolution or use a cloud GPU rather than changing the model silently.
- Training downloads gated/model artifacts from Hugging Face on first use and needs enough disk space for model weights and caches.
- MFLUX 0.24 exposes LoRA rank but not an independent training alpha. Its exported adapters have effective `alpha == rank` (scale 1.0). The project keeps alpha in YAML but rejects other values, so an experiment is never mislabeled.
- FLUX.2 distilled variants use guidance 1.0 and about 4 inference steps. Training targets the base variant, while the default inference config uses distilled `flux2-klein-4b`, matching upstream guidance that a base-trained adapter can be used with the faster distilled model.
- Reference conditioning is an edit operation, not a pure identity adapter. Prompts should explicitly ask the model to preserve the tent's construction and colors.
- The remote benchmark uses the official BFL submit/poll API shape. Pricing is configurable because API prices change; confirm the current values before a paid run.

Upstream references: [BFL/Hugging Face LoRA guide](https://huggingface.co/blog/black-forest-labs/flux-2-klein-lora) and [MFLUX](https://github.com/mflux-community/mflux).

## Setup

```bash
uv sync
cp .env.example .env
uv run product-lora --help
```

The first local generation or training run downloads model weights. If Hugging Face requires license acceptance or authentication, complete that before running MFLUX.

## Dataset

Put original photos directly in `data/tent/train/` and reserved photos in `data/tent/test/`. The tools only read these files. Each training image needs a same-stem UTF-8 caption:

```text
data/tent/train/front-daylight.jpg
data/tent/train/front-daylight.txt
```

Use contextual, content-focused captions with the identifier in every caption, for example:

```text
zkttent pitched on short grass in overcast daylight, front three-quarter view, eye-level camera
```

Vary background, lighting, distance, and angle. Avoid describing features that are not visible. Keep several photographs entirely out of training, including the benchmark reference image. Validate before training:

```bash
uv run product-lora validate-dataset \
  --path data/tent/train \
  --product-identifier zkttent
```

Validation checks supported formats (JPEG, PNG, WebP), image decoding, dimensions, missing/empty captions, identifier use, and byte-identical duplicate images. It never rewrites originals.

## Training

Every adjustable training value is in `configs/training.yaml`. `training_steps` is translated to MFLUX epochs using the validated image count and batch size. Because MFLUX stops at epoch boundaries, the requested step count must be a multiple of the batches per epoch; the command reports the valid multiple when it is not. `denoising_steps` is the separate diffusion schedule used by the base model. The command validates the dataset, creates a unique UTC run directory, copies the exact YAML, writes the generated MFLUX JSON and manifest, then launches MFLUX:

```bash
uv run product-lora train --config configs/training.yaml
```

Outputs are isolated under `outputs/training/train-<timestamp>/`. MFLUX writes checkpoint ZIPs, preview images, and an HTML loss plot there; the wrapper also captures the training log, extracts `adapter.safetensors` and `loss-history.json` from the final checkpoint, and copies the adapter to `outputs/adapters/<run-id>.safetensors`. A checkpoint is resumable with MFLUX directly:

```bash
uv run mflux-train --resume outputs/training/<run>/checkpoints/<checkpoint>.zip
```

Loss is useful for diagnosing instability, but choose a checkpoint using held-out visual samples; the lowest loss can be overfit.

## Generation and benchmark

Edit `configs/inference.yaml` to choose prompt, references, adapter, strength, resolution, and seeds:

```bash
uv run product-lora generate --config configs/inference.yaml
```

The benchmark config contains all five scenarios and these five comparisons:

1. Local prompt only
2. Local prompt + reference
3. Local prompt + LoRA
4. Local prompt + reference + LoRA
5. FLUX.2 Pro API prompt + reference

Set `adapter_path` and `reference_image`, add `BFL_API_KEY` to `.env`, verify the configured API pricing, then run:

```bash
uv run product-lora benchmark --config configs/benchmark.yaml
uv run product-lora report --run-id benchmark-YYYYMMDDTHHMMSSZ
```

The runner records each prompt, seed, path, elapsed time, and estimated API cost after every image, so a partial run remains inspectable. `report.html` lays configurations side by side for manual comparison. With the default five scenarios, five configurations, and three seeds, a full run produces 75 images, including 15 paid API calls.

## Design

- `config.py`: typed YAML contracts and compatibility guards
- `dataset.py`: read-only dataset validation
- `training.py`: transparent YAML-to-MFLUX config translation
- `models/local.py`: MFLUX command adapter
- `models/remote.py`: BFL API adapter
- `inference.py`: individual generation runs
- `benchmark.py`: experiment matrix, manifests, and HTML report

Run the fast tests without downloading model weights:

```bash
uv run pytest
uv run ruff check .
```

