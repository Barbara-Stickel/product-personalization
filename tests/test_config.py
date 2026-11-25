from pathlib import Path

import pytest
from pydantic import ValidationError

from product_lora.config import TrainingConfig, load_config


def test_example_training_config_is_valid() -> None:
    config = load_config(Path("configs/training.yaml"), TrainingConfig)
    assert config.model.name == "flux2-klein-base-4b"
    assert config.lora.alpha == config.lora.rank


def test_mflux_rejects_independent_alpha() -> None:
    config = load_config(Path("configs/training.yaml"), TrainingConfig).model_dump()
    config["lora"]["alpha"] = 8
    with pytest.raises(ValidationError, match="alpha equal to rank"):
        TrainingConfig.model_validate(config)
