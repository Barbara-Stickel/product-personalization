from __future__ import annotations

import json
import logging
import platform
import shutil
import subprocess
import zipfile
from datetime import UTC, datetime
from math import ceil
from pathlib import Path

from product_lora.config import TrainingConfig
from product_lora.dataset import validate_dataset

LOGGER = logging.getLogger(__name__)


def build_mflux_config(
    config: TrainingConfig, run_dir: Path, *, dataset_size: int
) -> dict[str, object]:
    batches_per_epoch = ceil(dataset_size / config.training.batch_size)
    if config.training.training_steps % batches_per_epoch:
        raise ValueError(
            "MFLUX stops at epoch boundaries. training_steps must be a multiple of "
            f"ceil(dataset_size / batch_size)={batches_per_epoch}; got "
            f"{config.training.training_steps}."
        )
    num_epochs = config.training.training_steps // batches_per_epoch
    rank = config.lora.rank
    targets = [
        {
            "module_path": f"transformer_blocks.{{block}}.{path}",
            "blocks": {"start": 0, "end": 5},
            "rank": rank,
        }
        for path in (
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out",
            "attn.add_q_proj",
            "attn.add_k_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.linear_in",
            "ff.linear_out",
            "ff_context.linear_in",
            "ff_context.linear_out",
        )
    ]
    targets.extend(
        {
            "module_path": f"single_transformer_blocks.{{block}}.{path}",
            "blocks": {"start": 0, "end": 20},
            "rank": rank,
        }
        for path in ("attn.to_qkv_mlp_proj", "attn.to_out")
    )
    return {
        "model": config.model.name,
        "model_path": config.model.model_path,
        "data": str(config.dataset.path.resolve()),
        "seed": config.training.seed,
        "steps": config.training.denoising_steps,
        "guidance": config.training.guidance,
        "quantize": config.model.quantization,
        "low_ram": config.training.low_ram,
        "cache_limit_gb": config.training.cache_limit_gb,
        "gradient_checkpointing": config.training.gradient_checkpointing,
        "max_resolution": config.training.max_resolution,
        "training_loop": {
            "num_epochs": num_epochs,
            "batch_size": config.training.batch_size,
            "timestep_low": config.training.timestep_low,
            "timestep_high": config.training.timestep_high,
        },
        "optimizer": {
            "name": config.optimizer.name,
            "learning_rate": config.optimizer.learning_rate,
        },
        "checkpoint": {
            # MFLUX appends a timestamp if its output path already exists.
            "output_path": str((run_dir / "mflux").resolve()),
            "save_frequency": config.checkpoint.frequency,
        },
        "monitoring": {
            "plot_frequency": config.monitoring.plot_frequency,
            "generate_image_frequency": config.monitoring.generate_image_frequency,
            "preview_width": config.monitoring.preview_width,
            "preview_height": config.monitoring.preview_height,
        },
        "lora_layers": {"targets": targets},
    }


def train(config: TrainingConfig, source_config: Path) -> str:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Local MFLUX training requires Apple Silicon (Darwin arm64)")
    if shutil.which("mflux-train") is None:
        raise RuntimeError("mflux-train is unavailable; run `uv sync` on an Apple Silicon Mac")
    report = validate_dataset(config.dataset.path, config.dataset.product_identifier)
    if not report.valid:
        raise ValueError("Dataset validation failed:\n" + "\n".join(report.errors))

    run_id = _run_id("train")
    run_dir = config.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source_config, run_dir / "config.yaml")
    mflux_config = build_mflux_config(config, run_dir, dataset_size=report.image_count)
    generated = run_dir / "mflux-config.json"
    generated.write_text(json.dumps(mflux_config, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "product_identifier": config.dataset.product_identifier,
        "mflux_version": "0.24.0",
        "note": "MFLUX embeds effective alpha=rank; independent alpha is not supported.",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    LOGGER.info("Starting MFLUX training run %s", run_id)
    with (run_dir / "training.log").open("w", encoding="utf-8") as log_file:
        subprocess.run(
            ["mflux-train", "--config", str(generated)],
            check=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    _export_final_adapter(run_dir, run_id, checkpoint_dir=run_dir / "mflux" / "checkpoints")
    return run_id


def _run_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"


def _export_final_adapter(run_dir: Path, run_id: str, *, checkpoint_dir: Path) -> None:
    checkpoints = sorted(checkpoint_dir.glob("*_checkpoint.zip"))
    if not checkpoints:
        raise RuntimeError(
            f"No MFLUX checkpoints found in {checkpoint_dir}. See {run_dir / 'training.log'}."
        )
    with zipfile.ZipFile(checkpoints[-1]) as archive:
        checkpoint = json.loads(archive.read("checkpoint.json"))
        adapter_member = checkpoint["files"]["lora_adapter"]
        adapter_bytes = archive.read(adapter_member)
        loss_member = checkpoint["files"]["loss"]
        loss_bytes = archive.read(loss_member)
    run_adapter = run_dir / "adapter.safetensors"
    run_adapter.write_bytes(adapter_bytes)
    (run_dir / "loss-history.json").write_bytes(loss_bytes)
    adapter_dir = Path("outputs/adapters")
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / f"{run_id}.safetensors").write_bytes(adapter_bytes)
