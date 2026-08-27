"""speedman command line.

Three commands -- `run`, `compare`, `doctor` -- but `run` is implicit, so
`speedman podcast.mp3` works without remembering a subcommand.
"""
from __future__ import annotations

import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Speech-aware time compression that stays intelligible.")

COMMANDS = {"run", "compare", "doctor"}


# --------------------------------------------------------------------------- ui

def _err(msg: str) -> None:
    typer.secho(msg, fg=typer.colors.RED, err=True)


def _ok(msg: str) -> None:
    typer.secho(msg, fg=typer.colors.GREEN)


def _dim(msg: str) -> None:
    typer.secho(msg, fg=typer.colors.BRIGHT_BLACK)


class Progress:
    """Plain status lines. Deliberately not a fancy progress bar: this has to be
    readable in a stock Windows terminal AND when piped to a file."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.t1 = None

    def __call__(self, msg: str) -> None:
        if not self.enabled:
            return
        self.finish()
        typer.echo(f"  {msg:<14} ", nl=False)
        self.t1 = time.perf_counter()

    def finish(self) -> None:
        if self.enabled and self.t1 is not None:
            typer.echo(f"{time.perf_counter() - self.t1:5.1f}s")
            self.t1 = None


def _open_folder(path: Path) -> None:
    try:
        if sys.platform == "win32":
            os.startfile(path)  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception:
        pass


# --------------------------------------------------------------------------- run

@app.command()
def run(
    infile: Path = typer.Argument(..., exists=True, dir_okay=False, help="Audio file to speed up"),
    out: Optional[Path] = typer.Option(None, "-o", "--out", help="Output file (default: alongside the input)"),
    speed: float = typer.Option(5.0, "--speed", "-s", min=1.01, max=30.0, help="How much faster"),
    preset: str = typer.Option("fast", "--preset", "-p", help="natural | fast | aggressive | max"),
    backend: str = typer.Option("rubberband", "--backend", "-b", hidden=True),
    staged: bool = typer.Option(False, "--staged/--no-staged", hidden=True),
    uniform: bool = typer.Option(False, "--uniform", help="Plain constant-rate stretch (what your player does)"),
    open_folder: bool = typer.Option(False, "--open", help="Open the output folder when finished"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
    debug: bool = typer.Option(False, "--debug", help="Show the full traceback on error"),
):
    """Speed up an audio file while keeping it intelligible."""
    from . import io as sio
    from .config import PRESETS, build_config
    from .pipeline import process

    if preset not in PRESETS:
        _err(f"'{preset}' is not a preset. Choose from: {', '.join(sorted(PRESETS))}")
        raise typer.Exit(2)

    cfg = build_config(speed=speed, preset=preset, backend=backend, staged=staged, uniform=uniform)
    out = out or infile.with_name(f"{infile.stem}_{speed:g}x.wav")

    try:
        y = sio.load(infile, cfg.sample_rate)
        dur = len(y) / cfg.sample_rate
        if not quiet:
            typer.echo(f"{infile.name}  {_hms(dur)} -> {_hms(dur / speed)}  "
                       f"({speed:g}x, {preset}{', uniform' if uniform else ''})")
        prog = Progress(enabled=not quiet)
        res = process(y, cfg.sample_rate, cfg, on_progress=prog)
        prog.finish()
        sio.save(out, res.audio, res.sr)
    except Exception as exc:
        if debug:
            raise
        _err(f"\n{type(exc).__name__}: {exc}")
        raise typer.Exit(1)

    if not quiet:
        n = res.notes
        if not uniform:
            _dim(f"  {n['silence_fraction']*100:.0f}% of the original is silence, so the speech "
                 f"itself runs at about {n['effective_speech_rate']:g}x")
        _ok(f"  wrote {out}")
        if abs(n["duration_error_pct"]) > 1.0:
            _dim(f"  (length is {n['duration_error_pct']:+.1f}% off target)")
    if open_folder:
        _open_folder(out.parent)


def _hms(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ----------------------------------------------------------------------- compare

@app.command()
def compare(
    infile: Path = typer.Argument(..., exists=True, dir_okay=False),
    speeds: str = typer.Option("5,6", "--speeds", help="Comma-separated, e.g. 4,5,6"),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir"),
    blind: bool = typer.Option(True, "--blind/--labelled",
                               help="Randomise filenames so labels do not bias what you hear"),
    open_folder: bool = typer.Option(True, "--open/--no-open"),
):
    """Render the same clip several ways so you can A/B them.

    Blind by default: the files get neutral names and the mapping is written to
    KEY.txt. Knowing which file is the clever one is the fastest way to convince
    yourself it sounds better when it does not.
    """
    from . import io as sio
    from .analyze import annotate_vad
    from .config import build_config
    from .pipeline import process

    try:
        speed_list = [float(s) for s in speeds.split(",") if s.strip()]
        if not speed_list:
            raise ValueError
    except ValueError:
        _err(f"could not read --speeds {speeds!r}; expected something like 5,6")
        raise typer.Exit(2)

    out_dir = out_dir or infile.with_name(f"{infile.stem}_compare")
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = [("uniform", dict(uniform=True)),
                ("natural", dict(preset="natural")),
                ("fast", dict(preset="fast")),
                ("aggressive", dict(preset="aggressive"))]

    y = sio.load(infile, 24000)
    dur = len(y) / 24000
    typer.echo(f"{infile.name}  {_hms(dur)}\n"
               f"rendering {len(speed_list) * len(variants)} versions "
               f"({len(variants)} variants x {len(speed_list)} speeds)\n")

    jobs, key = [], []
    for sp in speed_list:
        for label, kw in variants:
            jobs.append((sp, label, kw))
    names = [f"{chr(97 + i)}{j}" for j in range(len(speed_list)) for i in range(len(variants))]
    if blind:
        random.shuffle(names)

    sio.save(out_dir / "original.wav", y, 24000)
    typer.echo("  analysing once (shared by every variant) ...")
    ann = annotate_vad(y, 24000)

    for (sp, label, kw), name in zip(jobs, names):
        cfg = build_config(speed=sp, **{"preset": "fast", **kw})
        fname = f"{name}.wav" if blind else f"{label}_{sp:g}x.wav"
        typer.echo(f"  {label:<11} {sp:g}x -> {fname}")
        res = process(y, 24000, cfg, annotation=ann)
        sio.save(out_dir / fname, res.audio, res.sr)
        key.append((fname, label, sp, res.notes.get("effective_speech_rate", sp)))

    lines = ["speedman compare -- key", f"source: {infile.name}", ""]
    lines += [f"{f:<14} {lab:<11} {sp:g}x   (speech itself ~{eff:g}x)"
              for f, lab, sp, eff in sorted(key)]
    lines += ["", "'uniform' is the plain constant-rate stretch -- roughly what a media",
              "player's speed control gives you. That is the one to beat.", "",
              "Listen to original.wav first to re-set your ear, then the rest in a",
              "random order. Rate each for effort, not preference: how hard did you",
              "have to work to follow it?"]
    (out_dir / "KEY.txt").write_text("\n".join(lines), encoding="utf-8")

    _ok(f"\ndone -> {out_dir}")
    if blind:
        _dim("  filenames are randomised; KEY.txt has the mapping (don't peek first)")
    if open_folder:
        _open_folder(out_dir)


# ------------------------------------------------------------------------ doctor

@app.command()
def doctor():
    """Check that everything speedman needs is installed and working."""
    import shutil

    problems = []
    typer.echo("speedman setup check\n")

    v = sys.version_info
    _line("Python", f"{v.major}.{v.minor}.{v.micro}", v >= (3, 10),
          "speedman needs Python 3.10 or newer")

    for mod in ("numpy", "scipy", "soundfile", "librosa", "pyrubberband", "pyloudnorm"):
        try:
            __import__(mod)
            _line(mod, "installed", True, "")
        except ImportError:
            _line(mod, "MISSING", False, "")
            problems.append(f'{mod} is missing -- run:  pip install -e ".[dev]"')

    ff = shutil.which("ffmpeg")
    _line("ffmpeg", ff or "MISSING", bool(ff), "")
    if not ff:
        problems.append(
            "ffmpeg is missing. You need it for mp3/m4a/mp4 (WAV works without it).\n"
            "    Windows:  winget install ffmpeg\n"
            "    macOS:    brew install ffmpeg\n"
            "    Linux:    apt install ffmpeg")

    from .stretch import find_rubberband
    rb = find_rubberband()
    _line("rubberband", rb or "MISSING", bool(rb), "")
    if not rb:
        here = Path(__file__).resolve().parents[2] / "bin"
        problems.append(
            "rubberband is missing. This one is REQUIRED -- it is the only engine\n"
            "    that can apply a non-uniform time map, which is the whole point of\n"
            "    speedman. Without it you only get plain constant-rate stretching.\n\n"
            "    Easiest fix on Windows -- no PATH editing needed:\n"
            "      1. download the command-line utility (a zip) from\n"
            "         https://breakfastquay.com/rubberband/\n"
            "      2. open it and find rubberband.exe inside\n"
            f"      3. copy it into:  {here}\n\n"
            "    Or:  macOS  brew install rubberband\n"
            "         Linux  apt install rubberband-cli\n"
            "    Or set SPEEDMAN_RUBBERBAND to the full path of the binary.")

    if not problems:
        typer.echo("\nrunning a self-test on synthetic audio ...")
        try:
            ok, detail = _self_test()
            if ok:
                _ok(f"\nEverything works. {detail}")
                _dim("\nTry:  speedman compare yourfile.mp3")
                return
            _err(f"\nself-test failed: {detail}")
            raise typer.Exit(1)
        except Exception as exc:
            _err(f"\nself-test failed: {type(exc).__name__}: {exc}")
            raise typer.Exit(1)

    typer.echo("")
    n = len(problems)
    _err(f"{n} thing{'s' if n > 1 else ''} to fix:\n")
    for p in problems:
        typer.echo(f"  - {p}\n")
    raise typer.Exit(1)


def _line(name: str, value: str, ok: bool, _hint: str) -> None:
    mark = typer.style("OK  ", fg=typer.colors.GREEN) if ok else typer.style("FAIL", fg=typer.colors.RED)
    typer.echo(f"  {mark} {name:<14} {value}")


def _self_test() -> tuple[bool, str]:
    import numpy as np
    from scipy.signal import butter, sosfilt

    from .config import build_config
    from .pipeline import process

    sr, rng = 24000, np.random.default_rng(0)
    parts = []
    for _ in range(6):
        n = int(0.5 * sr)
        parts.append((sosfilt(butter(2, [200, 3000], "bp", fs=sr, output="sos"),
                              rng.normal(0, 1, n)) * 0.3).astype(np.float32))
        parts.append(np.zeros(int(0.35 * sr), np.float32))
    y = np.concatenate(parts)

    res = process(y, sr, build_config(speed=5.0))
    err = abs(res.notes["duration_error_pct"])
    peak = float(np.abs(res.audio).max())
    if err > 2.0:
        return False, f"length was {err:.2f}% off target"
    if peak > 0.9701:
        return False, f"output clipped (peak {peak:.3f})"
    if not np.isfinite(res.audio).all():
        return False, "output contained NaN or Inf"
    return True, f"(5x test: length {err:.2f}% off, peak {peak:.2f})"


# --------------------------------------------------------------------------- main

def main() -> None:
    """Entry point. Inserts the implicit `run` so `speedman file.mp3` works."""
    argv = sys.argv[1:]
    if argv and argv[0] not in COMMANDS and not argv[0].startswith("-"):
        sys.argv.insert(1, "run")
    app()


if __name__ == "__main__":
    main()
