from __future__ import annotations

import html
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from product_lora.config import BenchmarkConfig
from product_lora.models.base import GenerationRequest
from product_lora.models.local import MfluxGenerator
from product_lora.models.remote import BflGenerator


def run_benchmark(config: BenchmarkConfig, source_config: Path) -> str:
    _validate_assets(config)
    run_id = f"benchmark-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = config.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source_config, run_dir / "config.yaml")
    local = MfluxGenerator(config.local_model, config.local_quantization)
    remote = BflGenerator(config.remote)
    records: list[dict[str, object]] = []

    for scenario in config.scenarios:
        for variant in config.configurations:
            for index, seed in enumerate(config.seeds[: config.images_per_scenario], start=1):
                output = run_dir / scenario.id / variant.id / f"{index:02d}-seed-{seed}.png"
                request = _request(config, variant, scenario.prompt, output, seed)
                result = (remote if variant.backend == "remote" else local).generate(request)
                records.append(
                    {
                        "scenario": scenario.id,
                        "configuration": variant.id,
                        "prompt": scenario.prompt,
                        "path": str(output.relative_to(run_dir)),
                        "seed": seed,
                        "seconds": result.elapsed_seconds,
                        "api_cost_usd": result.api_cost_usd,
                    }
                )
                _write_manifest(run_dir, run_id, records)
    build_report(run_dir)
    return run_id


def _request(config, variant, prompt, output, seed) -> GenerationRequest:
    return GenerationRequest(
        prompt=prompt,
        output=output,
        seed=seed,
        width=config.width,
        height=config.height,
        steps=config.local_steps,
        guidance=config.local_guidance,
        reference_images=(config.reference_image,) if variant.reference else (),
        adapter_path=config.adapter_path if variant.lora else None,
        adapter_strength=config.adapter_strength,
    )


def _validate_assets(config: BenchmarkConfig) -> None:
    if any(v.reference for v in config.configurations) and not config.reference_image.is_file():
        raise FileNotFoundError(f"Reference image not found: {config.reference_image}")
    if any(v.lora for v in config.configurations) and not config.adapter_path.is_file():
        raise FileNotFoundError(f"LoRA adapter not found: {config.adapter_path}")


def _write_manifest(run_dir: Path, run_id: str, records: list[dict[str, object]]) -> None:
    payload = {"run_id": run_id, "updated_at": datetime.now(UTC).isoformat(), "images": records}
    (run_dir / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def build_report(run_dir: Path) -> Path:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    records = manifest["images"]
    scenarios = list(dict.fromkeys(item["scenario"] for item in records))
    variants = list(dict.fromkeys(item["configuration"] for item in records))
    sections = []
    for scenario in scenarios:
        cells = []
        for variant in variants:
            images = [
                r for r in records if r["scenario"] == scenario and r["configuration"] == variant
            ]
            figures = "".join(
                f'<figure><img src="{html.escape(r["path"])}" loading="lazy">'
                f"<figcaption>seed {r['seed']} · {r['seconds']:.1f}s"
                f"{' · $' + format(r['api_cost_usd'], '.3f') if r['api_cost_usd'] is not None else ''}</figcaption></figure>"
                for r in images
            )
            cells.append(f"<td><h3>{html.escape(variant)}</h3>{figures}</td>")
        sections.append(f"<h2>{html.escape(scenario)}</h2><table><tr>{''.join(cells)}</tr></table>")
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(manifest["run_id"])}</title>
<style>body{{font:15px system-ui;margin:2rem;background:#f5f5f2;color:#222}}table{{width:100%;table-layout:fixed;border-spacing:10px}}td{{vertical-align:top;background:white;padding:12px;border-radius:8px}}img{{width:100%;height:auto}}figure{{margin:0 0 16px}}figcaption{{color:#555;font-size:12px}}h2{{margin-top:3rem}}</style></head>
<body><h1>{html.escape(manifest["run_id"])}</h1>{"".join(sections)}</body></html>"""
    output = run_dir / "report.html"
    output.write_text(document, encoding="utf-8")
    return output
