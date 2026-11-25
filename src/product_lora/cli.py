from __future__ import annotations

import logging
from pathlib import Path

import typer
from dotenv import load_dotenv

from product_lora.benchmark import build_report, run_benchmark
from product_lora.config import BenchmarkConfig, InferenceConfig, TrainingConfig, load_config
from product_lora.dataset import validate_dataset as inspect_dataset
from product_lora.inference import generate as generate_images
from product_lora.training import train as train_model

app = typer.Typer(no_args_is_help=True, help="Train and benchmark product-specific FLUX.2 LoRAs.")


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)s %(message)s"
    )
    load_dotenv()


@app.command("validate-dataset")
def validate_dataset_command(
    path: Path = typer.Option(Path("data/tent/train"), exists=False),
    product_identifier: str = typer.Option("zkttent"),
) -> None:
    """Check image/caption pairs without modifying the original photographs."""
    report = inspect_dataset(path, product_identifier)
    for warning in report.warnings:
        typer.echo(f"WARNING: {warning}")
    for error in report.errors:
        typer.echo(f"ERROR: {error}", err=True)
    typer.echo(f"Checked {report.image_count} image(s)")
    if not report.valid:
        raise typer.Exit(1)


@app.command()
def train(config: Path = typer.Option(Path("configs/training.yaml"), exists=True)) -> None:
    """Validate the dataset, materialize an MFLUX config, and train a LoRA."""
    run_id = train_model(load_config(config, TrainingConfig), config)
    typer.echo(run_id)


@app.command()
def generate(config: Path = typer.Option(Path("configs/inference.yaml"), exists=True)) -> None:
    """Generate local FLUX.2 images, optionally with references and a LoRA."""
    run_id = generate_images(load_config(config, InferenceConfig))
    typer.echo(run_id)


@app.command()
def benchmark(config: Path = typer.Option(Path("configs/benchmark.yaml"), exists=True)) -> None:
    """Run the configured fidelity comparison and build its HTML report."""
    run_id = run_benchmark(load_config(config, BenchmarkConfig), config)
    typer.echo(run_id)


@app.command()
def report(
    run_id: str = typer.Option(...), output_root: Path = typer.Option(Path("outputs/benchmarks"))
) -> None:
    """Rebuild an HTML report from a completed or partial benchmark manifest."""
    path = build_report(output_root / run_id)
    typer.echo(path)
