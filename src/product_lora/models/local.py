from __future__ import annotations

import subprocess
import time

from product_lora.models.base import GenerationRequest, GenerationResult


class MfluxGenerator:
    def __init__(self, model: str, quantization: int | None) -> None:
        self.model = model
        self.quantization = quantization

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
        if self.quantization is not None:
            command.extend(["--quantize", str(self.quantization)])
        if request.reference_images:
            command.append("--image-paths")
            command.extend(str(path) for path in request.reference_images)
        if request.adapter_path:
            command.extend(
                [
                    "--lora-paths",
                    str(request.adapter_path),
                    "--lora-scales",
                    str(request.adapter_strength),
                ]
            )
        started = time.perf_counter()
        subprocess.run(command, check=True)
        return GenerationResult(request.output, time.perf_counter() - started)
