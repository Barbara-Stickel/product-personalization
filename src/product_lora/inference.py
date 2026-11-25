from __future__ import annotations

import json
from datetime import UTC, datetime

from product_lora.config import InferenceConfig
from product_lora.models.base import GenerationRequest
from product_lora.models.local import MfluxGenerator


def generate(config: InferenceConfig) -> str:
    run_id = f"generate-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = config.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    generator = MfluxGenerator(config.model.name, config.model.quantization)
    records = []
    for index, seed in enumerate(config.seeds, start=1):
        output = run_dir / f"image-{index:02d}-seed-{seed}.png"
        result = generator.generate(
            GenerationRequest(
                prompt=config.prompt,
                output=output,
                seed=seed,
                width=config.width,
                height=config.height,
                steps=config.steps,
                guidance=config.guidance,
                reference_images=tuple(config.reference_images),
                adapter_path=config.adapter_path,
                adapter_strength=config.adapter_strength,
            )
        )
        records.append({"path": str(output), "seed": seed, "seconds": result.elapsed_seconds})
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {"run_id": run_id, "config": config.model_dump(mode="json"), "images": records},
            indent=2,
        )
        + "\n"
    )
    return run_id
