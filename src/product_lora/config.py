from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeVar

import yaml
from pydantic import BaseModel, Field, model_validator

T = TypeVar("T", bound=BaseModel)


class ModelSettings(BaseModel):
    name: str
    quantization: int | None = Field(default=None, ge=3, le=8)


class DatasetSettings(BaseModel):
    path: Path
    product_identifier: str = Field(min_length=3, pattern=r"^[A-Za-z0-9_-]+$")


class LoraSettings(BaseModel):
    rank: int = Field(gt=0)
    alpha: int = Field(gt=0)


class OptimizerSettings(BaseModel):
    name: str = "AdamW"
    learning_rate: float = Field(gt=0)


class TrainingLoopSettings(BaseModel):
    training_steps: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    denoising_steps: int = Field(gt=0)
    guidance: float = Field(gt=0)
    max_resolution: int = Field(gt=0)
    seed: int
    timestep_low: int = Field(ge=0)
    timestep_high: int = Field(gt=0)
    gradient_checkpointing: bool = True
    low_ram: bool = True


class CheckpointSettings(BaseModel):
    frequency: int = Field(gt=0)


class MonitoringSettings(BaseModel):
    plot_frequency: int = Field(gt=0)
    generate_image_frequency: int = Field(gt=0)


class TrainingConfig(BaseModel):
    model: ModelSettings
    dataset: DatasetSettings
    lora: LoraSettings
    optimizer: OptimizerSettings
    training: TrainingLoopSettings
    checkpoint: CheckpointSettings
    monitoring: MonitoringSettings
    output_root: Path

    @model_validator(mode="after")
    def validate_mflux_compatibility(self) -> TrainingConfig:
        if self.model.name != "flux2-klein-base-4b":
            raise ValueError("This experiment intentionally trains flux2-klein-base-4b only")
        if self.lora.alpha != self.lora.rank:
            raise ValueError(
                "MFLUX 0.24 does not expose independent training alpha; set alpha equal to rank"
            )
        if self.training.timestep_low >= self.training.timestep_high:
            raise ValueError("timestep_low must be lower than timestep_high")
        return self


class InferenceConfig(BaseModel):
    backend: Literal["local"] = "local"
    model: ModelSettings
    prompt: str = Field(min_length=1)
    reference_images: list[Path] = []
    adapter_path: Path | None = None
    adapter_strength: float = Field(default=1.0, ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    steps: int = Field(gt=0)
    guidance: float = Field(gt=0)
    seeds: list[int] = Field(min_length=1)
    output_root: Path


class Scenario(BaseModel):
    id: str
    prompt: str


class BenchmarkVariant(BaseModel):
    id: str
    backend: Literal["local", "remote"]
    reference: bool
    lora: bool

    @model_validator(mode="after")
    def remote_cannot_use_local_lora(self) -> BenchmarkVariant:
        if self.backend == "remote" and self.lora:
            raise ValueError("Remote FLUX.2 Pro cannot load a local MFLUX adapter")
        return self


class RemoteSettings(BaseModel):
    endpoint: str
    api_key_env: str
    cost_per_output_megapixel: float = Field(ge=0)
    cost_per_input_megapixel: float = Field(ge=0)
    timeout_seconds: int = Field(default=600, gt=0)
    poll_interval_seconds: float = Field(default=1.0, gt=0)


class BenchmarkConfig(BaseModel):
    run_name: str
    output_root: Path
    images_per_scenario: int = Field(gt=0)
    seeds: list[int] = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    local_model: str
    local_quantization: int | None = None
    local_steps: int = Field(gt=0)
    local_guidance: float = Field(gt=0)
    adapter_path: Path
    adapter_strength: float = Field(ge=0)
    reference_image: Path
    remote: RemoteSettings
    scenarios: list[Scenario] = Field(min_length=1)
    configurations: list[BenchmarkVariant] = Field(min_length=1)

    @model_validator(mode="after")
    def seeds_cover_requested_images(self) -> BenchmarkConfig:
        if len(self.seeds) < self.images_per_scenario:
            raise ValueError("seeds must contain at least images_per_scenario values")
        return self


def load_config(path: Path, model: type[T]) -> T:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return model.model_validate(raw)
