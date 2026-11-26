from pathlib import Path
from unittest.mock import patch

from product_lora.config import InferenceConfig, load_config
from product_lora.inference import generate


def test_inference_config_passes_prequantized_base_and_adapter_flags(tmp_path: Path) -> None:
    config = load_config(Path("configs/inference.yaml"), InferenceConfig)
    config.output_root = tmp_path

    with patch("product_lora.models.local.subprocess.run") as run:
        generate(config)

    command = run.call_args.args[0]
    assert command[0] == "mflux-generate-flux2"
    assert command[command.index("--model") + 1] == config.model.name
    assert command[command.index("--base-model") + 1] == "flux2-klein-base-4b"
    assert command[command.index("--lora") + 1 : command.index("--lora") + 3] == [
        str(config.adapter_path),
        "1.0",
    ]
    assert {"--no-bake-lora", "--low-ram"} <= set(command)
    assert "--quantize" not in command
