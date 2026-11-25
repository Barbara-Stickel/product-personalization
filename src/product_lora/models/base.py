from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    output: Path
    seed: int
    width: int
    height: int
    steps: int
    guidance: float
    reference_images: tuple[Path, ...] = ()
    adapter_path: Path | None = None
    adapter_strength: float = 1.0


@dataclass(frozen=True)
class GenerationResult:
    output: Path
    elapsed_seconds: float
    api_cost_usd: float | None = None


class Generator(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResult: ...
