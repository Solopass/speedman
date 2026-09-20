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

### Running the Listening Test
The only instrument that reaches 5-6x. Open http://127.0.0.1:8081 -> **Listening Test**,
run a session, then pool it:
```bash
wsl --cd /mnt/d/Workspace/speedman .venv/bin/python -m eval.listening_report
```
Blinding is enforced server-side (slot-addressed audio, `/results` 409s until complete) —
don't move the slot→condition mapping into a client payload.

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

## YouTube / URL ingest

`POST /api/v1/ingest/url` (Studio -> "YouTube / Web URL") **delegates the download to
Media API**, which owns the auto-updating yt-dlp this workstation uses. There is no
yt-dlp in WSL or in the venv, and installing a second one is the wrong fix: YouTube
breaks extractors constantly and a private copy would rot silently until the day it was
needed. If Media API is down, the error says so and names the two ways out.

The download runs **inside the job**, not inside the request — a long podcast would
otherwise hold the HTTP connection open for minutes. Clients poll `/api/v1/jobs/{id}`
and see a `downloading` stage before `stretching`.

Tick **"also download the video"** and the job fetches video instead of audio-only and
stages it in `D:\Audio\Speed\video-cache\`; `sio.load` decodes the audio straight out of
the mp4, so there is no second download. The player offers Save (moves it to
`D:\Output\Videos`) and Close (deletes it). **Default is delete; audio and transcripts are
never touched.**

`app/video.py` is the only code in the repo that deletes user-visible files. Its rules:
deletion is confined to the cache directory and re-checked at the moment of unlinking;
everything in the cache is Speedman's own copy, so ownership is settled by moving/copying
the file in rather than by bookkeeping at delete time; a video the library already had is
copied (not moved) and `pre_existed` then makes *saving* a no-op. Startup sweeps unsaved
entries older than 24h. Do not relax any of this — `D:\Output\Videos` is a real library.

## Invariants worth not regressing

- **`ALLOWED_ROOTS` gates compression *inputs*, never HTTP *outputs*.** It spans `D:\Workspace`, `D:\AI` and `D:\OBVLT`, so validating a served path against it is a directory traversal. Anything reachable over HTTP goes through `resolve_output_file()`, which is confined to `OUTPUT_DIR`.
- **CORS is not a wildcard.** The Studio is same-origin and needs no grant; a wildcard only hands this loopback service (file reads, `/api/v1/shutdown`) to whatever page the user is browsing. Widen via `SPEEDMAN_CORS_ORIGINS` if a real client needs it.
- **User-supplied URLs never reach yt-dlp's option parser** — scheme-checked, and passed after a `--` terminator.
- **Speed-synced transcripts.** `POST /api/v1/transcribe/sync` warps a 1x transcript onto a compressed output's timeline and writes `<stem>.vtt` (a subtitle track players load natively) and `<stem>.synced.json`. It needs the time map, which `app/timemap_store.py` records on every compression — anything produced before that existed has none. Naive `t / N` drifts with file length: 0.2s on a 90-second clip, **4.2s across a 35-minute podcast**. The `/sync` route MUST stay declared above `/api/v1/transcribe/{filename:path}` — that parameter is greedy and silently swallowed it once.
- **Transcription runs on the SOURCE, never on Speedman's output.** Measured on a 45s clip: parakeet returns 150 coherent words at 1x and 11 words of nonsense at 5x. ASR engines are not trained on time-compressed speech and collapse exactly where Speedman becomes useful. Compression results carry `source_path` for this; `/api/v1/transcribe` refuses a path that looks like one of our outputs rather than returning a useless transcript.
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
