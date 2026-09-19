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
All 51 unit & API tests must pass.

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

## Code Structure

- `app/api.py`: FastAPI server (health, compress, compare, comparisons, audio streaming, heartbeat, watchdog)
- `app/static/index.html`: Web Studio Cockpit (single file compression & blind A/B evaluation player)
- `app/range_response.py`: HTTP 206 Partial Content range request handler for audio scrubbing
- `src/speedman/analyze.py`: VAD & onset/transient analysis
- `src/speedman/ratemap.py`: Non-uniform time map generation with boundary pause compression
- `src/speedman/stretch.py`: Rubberband backend integration
- `src/speedman/post.py`: Post-stretch clarity DSP chain (tilt EQ, dynamic compression, limiter)
- `src/speedman/pipeline.py`: Orchestrator (load -> analyze -> ratemap -> stretch -> post -> save)
- `src/speedman/cli.py`: Command-line interface (`speedman run`, `compare`, `doctor`)
