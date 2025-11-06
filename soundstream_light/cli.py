"""Typer-based entry point for the SoundStream tooling."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import models
from .runtime import (
    ENV_BINARY,
    ENV_MODELS,
    locate_binary,
    resolve_models_dir,
    run_cli,
)

console = Console()
app = typer.Typer(help="SoundStream embedding utilities")
models_app = typer.Typer(help="Manage SoundStream model assets")
app.add_typer(models_app, name="models")

@app.command()
def encode(
    wav: Path = typer.Argument(..., exists=True, readable=True, help="Input WAV"),
    embeddings: Path = typer.Argument(..., help="Output embeddings path"),
    models_dir: Path | None = typer.Option(None, help="Model directory (defaults to auto-detection)"),
    threads: int = typer.Option(1, min=1, help="TFLite thread count"),
    use_xnnpack: bool = typer.Option(True, help="Enable XNNPACK delegate"),
    binary: Path | None = typer.Option(None, help="Path to the native soundstream_cli binary"),
) -> None:
    """Encode WAV audio into SoundStream embeddings."""
    resolved_binary, message = locate_binary(binary)
    if resolved_binary is None:
        console.print(message or "Native binary not found.", style="red")
        raise typer.Exit(1)

    resolved_models = resolve_models_dir(models_dir)
    embeddings.parent.mkdir(parents=True, exist_ok=True)
    resolved_models.mkdir(parents=True, exist_ok=True)

    args = [
        "encode",
        "--wav",
        str(wav),
        "--embeddings",
        str(embeddings),
        "--models",
        str(resolved_models),
    ]
    if threads != 1:
        args.extend(["--threads", str(threads)])
    if not use_xnnpack:
        args.append("--no-xnn")

    exit_code = run_cli(resolved_binary, args)
    raise typer.Exit(exit_code)


@app.command()
def decode(
    embeddings: Path = typer.Argument(..., exists=True, readable=True, help="Input embedding file"),
    wav: Path = typer.Argument(..., help="Output WAV path"),
    models_dir: Path | None = typer.Option(None, help="Model directory (defaults to auto-detection)"),
    threads: int = typer.Option(1, min=1, help="TFLite thread count"),
    use_xnnpack: bool = typer.Option(True, help="Enable XNNPACK delegate"),
    binary: Path | None = typer.Option(None, help="Path to the native soundstream_cli binary"),
) -> None:
    """Decode SoundStream embeddings back into audio."""
    resolved_binary, message = locate_binary(binary)
    if resolved_binary is None:
        console.print(message or "Native binary not found.", style="red")
        raise typer.Exit(1)

    resolved_models = resolve_models_dir(models_dir)
    wav.parent.mkdir(parents=True, exist_ok=True)
    resolved_models.mkdir(parents=True, exist_ok=True)

    args = [
        "decode",
        "--embeddings",
        str(embeddings),
        "--wav",
        str(wav),
        "--models",
        str(resolved_models),
    ]
    if threads != 1:
        args.extend(["--threads", str(threads)])
    if not use_xnnpack:
        args.append("--no-xnn")

    exit_code = run_cli(resolved_binary, args)
    raise typer.Exit(exit_code)


@models_app.command("fetch")
def models_fetch(
    model_dir: Path | None = typer.Option(None, help="Destination directory"),
    base_url: str = typer.Option(models.DEFAULT_BASE_URL, help="Base URL for downloads"),
    overwrite: bool = typer.Option(False, help="Overwrite existing models"),
    local_source: Path | None = typer.Option(None, help="Copy models from a local directory"),
) -> None:
    """Download or copy the required TFLite model files."""
    resolved_models = resolve_models_dir(model_dir)
    report = models.download_models(resolved_models, base_url, overwrite, local_source)
    table = Table(title="Model Download Summary")
    table.add_column("Status")
    table.add_column("Files")
    table.add_row("fetched", "\n".join(str(path) for path in report.fetched) or "-")
    table.add_row("skipped", "\n".join(str(path) for path in report.skipped) or "-")
    table.add_row("failed", "\n".join(str(path) for path in report.failed) or "-")
    console.print(table)
    exit_code = 0 if not report.failed else 1
    raise typer.Exit(exit_code)


@models_app.command("verify")
def models_verify(
    model_dir: Path | None = typer.Option(None, help="Directory to verify"),
) -> None:
    """Verify model checksums in the specified directory."""
    resolved_models = resolve_models_dir(model_dir)
    results = models.verify_models(resolved_models)
    table = Table(title="Model Verification")
    table.add_column("File")
    table.add_column("Status")
    for filename, ok in results.items():
        table.add_row(filename, "ok" if ok else "missing or invalid")
    console.print(table)
    exit_code = 0 if all(results.values()) else 1
    raise typer.Exit(exit_code)

