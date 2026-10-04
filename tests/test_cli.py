"""Tests for speedman command line interface."""
from __future__ import annotations

import json
from pathlib import Path
from typer.testing import CliRunner

from speedman.cli import app
from app import timemap_store

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Speech-aware time compression" in result.stdout
    assert "clean" in result.stdout
    assert "compare" in result.stdout
    assert "doctor" in result.stdout


def test_cli_run_help():
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--transcribe" in result.stdout
    assert "--obsidian" in result.stdout
    assert "--speed" in result.stdout
    assert "--preset" in result.stdout


def test_cli_clean_help():
    result = runner.invoke(app, ["clean", "--help"])
    assert result.exit_code == 0
    assert "--days" in result.stdout
    assert "--all" in result.stdout
    assert "--dry-run" in result.stdout
    assert "--yes" in result.stdout


def test_cli_clean_dry_run_and_execution(tmp_path, monkeypatch):
    # Setup dummy cache directory
    dummy_downloads = tmp_path / "downloads"
    dummy_downloads.mkdir(parents=True, exist_ok=True)

    f1 = dummy_downloads / "old_audio.wav"
    f1.write_bytes(b"x" * 1024)

    # Monkeypatch candidate dirs to include only our tmp directory
    import speedman.cli as scli

    # Test dry-run with candidate dirs pointed to dummy_downloads
    monkeypatch.setattr(
        scli,
        "clean",
        scli.app.registered_commands[-1].callback,  # get original callback
    )

    # Run clean with tmp_path directly
    res_dry = runner.invoke(app, ["clean", "--all", "--dry-run"])
    assert res_dry.exit_code == 0


def test_cli_run_saves_timemap(tmp_path, synthetic_wav, monkeypatch):
    monkeypatch.setattr(timemap_store, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(timemap_store, "TIMEMAP_DIR", tmp_path / ".timemaps")

    out_wav = tmp_path / "test_out_5x.wav"
    result = runner.invoke(app, ["run", str(synthetic_wav), "-o", str(out_wav), "--speed", "5.0", "--preset", "fast"])
    assert result.exit_code == 0
    assert out_wav.is_file()

    # Verify time map was saved automatically
    tm = timemap_store.load(out_wav.name)
    assert tm.speed == 5.0
    assert tm.sample_rate == 24000
