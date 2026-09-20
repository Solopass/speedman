# Speedman — Developer & Machine Guide

Speedman is an on-demand, speech-aware time compression microservice that makes high-speed speech (5×–6×) intelligible by compressing pauses harder than speech and reallocating time budgets from steady-state vowels to consonant transients.

## Quick Facts

- **Repo**: `D:\Workspace\speedman` (git repo pushing to `github.com/Solopass/speedman`)
- **Port**: `127.0.0.1:8081` (WSL socket-activated `speedman.socket` -> `speedman.service`)
- **Core allocation**: Pinned to E-cores (Cores 8–15, `nice -n 10`, `ionice -c 3`)
- **Standby memory**: **0 MB RAM** on standby (starts via socket activation on first request, shuts down after 60s idle when client windows close)
- **Output directory**: `D:\Audio\Speed` (`/mnt/d/Audio/Speed`) — 0-leak policy on `C:\`
- **License**: PolyForm Noncommercial 1.0.0

## Running & Testing

### Running Tests
Inside WSL:
```bash
wsl --cd /mnt/d/Workspace/speedman .venv/bin/python -m pytest
```
All 75 unit & API tests must pass (3 are skipped when optional backends are absent).

### Running the Evaluation Harness
Needs Media API up on `127.0.0.1:8080` (it supplies the ASR engine):
```bash
wsl --cd /mnt/d/Workspace/speedman .venv/bin/python -m eval.harness
```
Writes `eval/results.md` — the file `config.py` defers to for every DSP constant. Results
cache in `eval/cache.json`; pass `--no-cache` after changing anything in `src/speedman/`,
because the cache cannot see code changes. **The ASR metric saturates above ~3.5x and
cannot evaluate the 5-6x target band** — see `eval/README.md` before reading a number.

### Running Service Standalone
```bash
wsl --cd /mnt/d/Workspace/speedman ./run.sh
```

### Windows Desktop & Explorer Integration
- Desktop app launcher: `D:\Workspace\speedman\Speedman-Silent.vbs` (or Desktop shortcut `Speedman.lnk`)
- Console launcher: `D:\Workspace\speedman\Speedman.bat`
- Shutdown script: `D:\Workspace\speedman\Stop-Speedman.bat`
- Explorer Send-To: Right-click audio file -> `Send to` -> `Speedman (5x Fast)` (`speedman-sendto.ps1`)
- Rebuild shortcuts: `powershell -File Create-Desktop-Shortcut.ps1`

## Invariants worth not regressing

- **`ALLOWED_ROOTS` gates compression *inputs*, never HTTP *outputs*.** It spans `D:\Workspace`, `D:\AI` and `D:\OBVLT`, so validating a served path against it is a directory traversal. Anything reachable over HTTP goes through `resolve_output_file()`, which is confined to `OUTPUT_DIR`.
- **CORS is not a wildcard.** The Studio is same-origin and needs no grant; a wildcard only hands this loopback service (file reads, `/api/v1/shutdown`) to whatever page the user is browsing. Widen via `SPEEDMAN_CORS_ORIGINS` if a real client needs it.
- **User-supplied URLs never reach yt-dlp's option parser** — scheme-checked, and passed after a `--` terminator.
- **Output formats come from `speedman.io.normalize_output_format`**, which raises rather than falling back to wav; a silent fallback writes a file whose extension contradicts the API response.

## Code Structure

- `app/api.py`: FastAPI server (health, compress, compare, comparisons, audio streaming, heartbeat, watchdog)
- `app/static/index.html`: Web Studio Cockpit (single file compression & blind A/B evaluation player)
- `app/range_response.py`: HTTP 206 Partial Content range request handler for audio scrubbing
- `app/queue.py`: Background job worker with progress, measured ETA, and cancellation
- `app/media_api.py`: Media API (`127.0.0.1:8080`) client, yt-dlp URL ingest, local library scan
- `app/paths.py`: Windows <-> WSL path translation and containment checks, shared by the above
- `src/speedman/analyze.py`: VAD & onset/transient analysis
- `src/speedman/chunking.py`: Pause-aligned chunking for long files (progress + cancellation)
- `src/speedman/ratemap.py`: Non-uniform time map generation with boundary pause compression
- `src/speedman/stretch.py`: Rubberband backend integration
- `src/speedman/post.py`: Post-stretch clarity DSP chain (tilt EQ, dynamic compression, limiter)
- `src/speedman/pipeline.py`: Orchestrator (load -> analyze -> ratemap -> stretch -> post -> save)
- `src/speedman/cli.py`: Command-line interface (`speedman run`, `compare`, `doctor`)
- `eval/`: Evaluation harness — WER scoring (`eval.harness`) and parameter sweeps (`eval.sweep`)
- `docs/EVALUATION.md`: What the evaluation has proven, ruled out, and left unproven — **read before changing any DSP constant**
