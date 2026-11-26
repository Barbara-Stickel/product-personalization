from __future__ import annotations

import subprocess
import time

from product_lora.models.base import GenerationRequest, GenerationResult


class MfluxGenerator:
    def __init__(
        self,
        model: str,
        quantization: int | None,
        *,
        base_model: str | None = None,
        bake_lora: bool = True,
        low_ram: bool = False,
    ) -> None:
        self.model = model
        self.quantization = quantization
        self.base_model = base_model
        self.bake_lora = bake_lora
        self.low_ram = low_ram

    def generate(self, request: GenerationRequest) -> GenerationResult:
        request.output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "mflux-generate-flux2-edit" if request.reference_images else "mflux-generate-flux2",
            "--model",
            self.model,
            "--prompt",
            request.prompt,
            "--steps",
            str(request.steps),
            "--seed",
            str(request.seed),
            "--width",
            str(request.width),
            "--height",
            str(request.height),
            "--guidance",
            str(request.guidance),
            "--output",
            str(request.output),
        ]
        if self.base_model is not None:
            command.extend(["--base-model", self.base_model])
        if self.low_ram:
            command.append("--low-ram")
        if self.quantization is not None:
            command.extend(["--quantize", str(self.quantization)])
        if request.reference_images:
            command.append("--image-paths")
            command.extend(str(path) for path in request.reference_images)
        if request.adapter_path:
            command.extend(
                [
                    "--lora",
                    str(request.adapter_path),
                    str(request.adapter_strength),
                ]
            )
            if not self.bake_lora:
                command.append("--no-bake-lora")
        started = time.perf_counter()
        subprocess.run(command, check=True)
        return GenerationResult(request.output, time.perf_counter() - started)
