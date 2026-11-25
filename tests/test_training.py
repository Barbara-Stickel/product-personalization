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


def test_final_checkpoint_exports_adapter_and_loss(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "0000010_checkpoint.zip"
    metadata = {"files": {"lora_adapter": "0000010_adapter.safetensors", "loss": "loss.json"}}
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("checkpoint.json", json.dumps(metadata))
        archive.writestr("0000010_adapter.safetensors", b"adapter")
        archive.writestr("loss.json", b"[]")

    _export_final_adapter(tmp_path / "run", "train-test")

    assert (tmp_path / "run" / "adapter.safetensors").read_bytes() == b"adapter"
    assert (tmp_path / "run" / "loss-history.json").read_bytes() == b"[]"
    assert (tmp_path / "outputs/adapters/train-test.safetensors").read_bytes() == b"adapter"
