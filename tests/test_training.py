import json
import zipfile
from pathlib import Path

from product_lora.config import TrainingConfig, load_config
from product_lora.training import _export_final_adapter, build_mflux_config


def test_training_translation_keeps_adjustable_values(tmp_path: Path) -> None:
    source = load_config(Path("configs/training.yaml"), TrainingConfig)
    translated = build_mflux_config(source, tmp_path, dataset_size=30)
    assert translated["model"] == "flux2-klein-base-4b"
    assert translated["quantize"] == 8
    assert translated["training_loop"]["batch_size"] == 1
    assert translated["training_loop"]["num_epochs"] == 40
    assert translated["lora_layers"]["targets"][0]["rank"] == 16
    output_path = Path(translated["checkpoint"]["output_path"])
    assert output_path.parent == tmp_path.resolve()
    assert not output_path.exists()


def test_smoke_config_uses_checkpoint_precision_and_smaller_previews(tmp_path: Path) -> None:
    config = load_config(Path("configs/training-smoke.yaml"), TrainingConfig)
    translated = build_mflux_config(config, tmp_path, dataset_size=24)

    assert translated["model"] == "flux2-klein-base-4b"
    assert translated["model_path"] == "mflux-community/flux-2-klein-base-4b-mflux-q4"
    assert translated["quantize"] is None
    assert translated["cache_limit_gb"] == 0.5
    assert translated["max_resolution"] == 512
    assert translated["monitoring"]["preview_width"] == 512
    assert translated["monitoring"]["preview_height"] == 512
    assert translated["training_loop"]["num_epochs"] == 10
    assert translated["checkpoint"]["save_frequency"] == 24


def test_final_checkpoint_exports_adapter_and_loss(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    checkpoint_dir = tmp_path / "run" / "mflux" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "0000010_checkpoint.zip"
    metadata = {"files": {"lora_adapter": "0000010_adapter.safetensors", "loss": "loss.json"}}
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("checkpoint.json", json.dumps(metadata))
        archive.writestr("0000010_adapter.safetensors", b"adapter")
        archive.writestr("loss.json", b"[]")

    _export_final_adapter(tmp_path / "run", "train-test", checkpoint_dir=checkpoint_dir)

    assert (tmp_path / "run" / "adapter.safetensors").read_bytes() == b"adapter"
    assert (tmp_path / "run" / "loss-history.json").read_bytes() == b"[]"
    assert (tmp_path / "outputs/adapters/train-test.safetensors").read_bytes() == b"adapter"
