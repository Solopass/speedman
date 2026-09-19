# Speedman API Reference & Architecture Guide

Speedman is an on-demand, speech-aware time compression microservice that makes speech intelligible at **5×–6× speeds**.

- **Host & Port:** `http://127.0.0.1:8081` (Loopback only)
- **Interactive Swagger Docs:** `http://127.0.0.1:8081/docs`
- **Output Storage:** `D:\Audio\Speed\` (`/mnt/d/Audio/Speed/`)
- **Core Allocation:** Pinned to workstation E-cores (Cores 8–15, `nice -n 10`, `ionice -c 3`)
- **Memory Footprint:** Zero-RAM standby via systemd socket activation; auto-shuts down after 60s idle when client windows close.

---

## 🧭 Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web Studio Cockpit (HTML dashboard with single & blind A/B evaluation) |
| `GET` | `/health` | Service health, version, active jobs, active clients, output directory, and Media API connectivity |
| `GET` | `/docs` | OpenAPI / Swagger interactive documentation |
| `GET` | `/api/v1/presets` | List DSP presets (`natural`, `fast`, `aggressive`, `max`) and ratemap parameters |
| `POST` | `/api/v1/compress` | Multipart form audio compression (file upload or workstation path) |
| `POST` | `/api/v1/compress/json` | JSON body synchronous audio compression for workstation paths |
| `POST` | `/api/v1/jobs/compress` | Queue non-blocking asynchronous compression job on E-cores (returns `job_id`) |
| `GET` | `/api/v1/jobs/{job_id}` | Poll real-time progress %, chunk count, stage, and measured ETA |
| `POST` | `/api/v1/jobs/{job_id}/cancel` | Cancel queued or active compression job, immediately terminating processing |
| `GET` | `/api/v1/jobs` | List recent background compression jobs |
| `GET` | `/api/v1/media/status` | Probe Media API (127.0.0.1:8080) health and connectivity |
| `GET` | `/api/v1/media/library` | Browse media available in Media API output directories (`D:\Output\Audio\`) |
| `POST` | `/api/v1/ingest/url` | Extract audio from YouTube or web media URL via yt-dlp and queue Speedman compression |
| `POST` | `/api/v1/transcribe/{filename:path}` | Forward sped-up audio to Media API for Parakeet Speech-To-Text transcription |
| `POST` | `/api/v1/compare` | Generate full randomized blind A/B comparison set with `KEY.txt` and `key.json` |
| `GET` | `/api/v1/comparisons` | List recent comparison evaluation sets |
| `GET` | `/api/v1/comparisons/{folder_id}/key` | Reveal the key mapping for a blind evaluation set |
| `GET` | `/api/v1/audio/{filename:path}` | Stream compressed audio with HTTP 206 Partial Content range support |
| `POST` | `/api/v1/heartbeat` | Keep-alive heartbeat from active client tabs/windows |
| `POST` | `/api/v1/client-disconnect` | Notification that a client tab/window closed (triggers 60s idle countdown) |
| `POST` | `/api/v1/shutdown` | Graceful service termination (releases all RAM and CPU threads) |

---

## 🛠️ Endpoint Specifications

### 1. `GET /health`
Returns service status, telemetry, and paths.

```json
{
  "status": "ok",
  "service": "speedman",
  "version": "1.0.0",
  "active_jobs": 0,
  "active_clients": 1,
  "idle_seconds": 12,
  "output_dir": "/mnt/d/Audio/Speed",
  "windows_output_dir": "D:\\Audio\\Speed"
}
```

---

### 2. `POST /api/v1/compress/json`
Fast programmatic compression for audio already on the workstation.

#### Request Body:
```json
{
  "input_path": "D:\\Audio\\Speed\\lecture.mp3",
  "speed": 5.0,
  "preset": "fast",
  "uniform": false
}
```

#### Response:
```json
{
  "status": "success",
  "filename": "lecture_5x_fast.wav",
  "output_path": "/mnt/d/Audio/Speed/lecture_5x_fast.wav",
  "windows_output_path": "D:\\Audio\\Speed\\lecture_5x_fast.wav",
  "audio_url": "/api/v1/audio/lecture_5x_fast.wav",
  "speed": 5.0,
  "preset": "fast",
  "uniform": false,
  "input_duration_s": 600.0,
  "output_duration_s": 120.0,
  "compression_ratio": 5.0,
  "silence_fraction": 0.28,
  "effective_speech_rate": 4.16,
  "processing_time_s": 14.32,
  "timings": {
    "analyze_s": 2.1,
    "ratemap_s": 0.05,
    "stretch_s": 11.2,
    "post_s": 0.97
  }
}
```

---

### 3. `POST /api/v1/compress` (Multipart Upload)
Compresses an uploaded audio file.

- `file`: Multipart file upload (`.wav`, `.mp3`, `.m4a`, `.aac`, `.flac`, `.ogg`, `.opus`).
- `speed`: Float (default `5.0`).
- `preset`: String (`natural`, `fast`, `aggressive`, `max`).
- `uniform`: Boolean (default `false`).

---

### 4. `POST /api/v1/compare`
Renders the clip across multiple speeds (e.g. `4.0,5.0,6.0`) and presets (`natural`, `fast`, `aggressive`, plus uniform stretch control).
All output tracks are assigned randomized names (`track_01.wav`, `track_02.wav`...) and mapped in `key.json` and `KEY.txt`.

#### Request (Form):
- `input_path` or `file`: Audio source.
- `speeds_str`: Comma-separated speeds (e.g. `"4.0,5.0,6.0"`).

#### Response:
```json
{
  "status": "success",
  "folder_id": "20260919_184500_sample",
  "folder_path": "/mnt/d/Audio/Speed/comparisons/20260919_184500_sample",
  "windows_folder_path": "D:\\Audio\\Speed\\comparisons\\20260919_184500_sample",
  "original_url": "/api/v1/audio/comparisons/20260919_184500_sample/original.wav",
  "original_duration_s": 45.2,
  "tracks": [
    {
      "track_id": "track_01.wav",
      "url": "/api/v1/audio/comparisons/20260919_184500_sample/track_01.wav",
      "duration_s": 9.04
    }
  ],
  "total_variants": 12,
  "processing_time_s": 8.42
}
```

---

### 5. `GET /api/v1/comparisons/{folder_id}/key`
Fetches the underlying mapping for a blind evaluation set:

```json
{
  "track_01.wav": {
    "label": "5x_fast",
    "speed": 5.0,
    "preset": "fast",
    "uniform": false,
    "duration_s": 9.04,
    "effective_speech_rate": 4.2
  }
}
```

---

## 🎛️ DSP Presets Explained

| Preset | Pause Compression | Vowel Budget Cut | Transient Protection | Clarity Chain | Use Case |
|---|---|---|---|---|---|
| **`natural`** | Light (0.75) | Minimal | High | Mild high-pass & gentle peak limiting | Casual speech, audiobooks, where acoustic naturalness is prioritized. |
| **`fast`** *(Default)* | Moderate (0.60) | Balanced | Very High | EQ presence boost + transient preservation | Podcasts, lectures, meetings. The sweet spot for 5× listening. |
| **`aggressive`** | Heavy (0.45) | Aggressive | Maximum | Strong presence tilt + formant punch | Dense informational audio, rapid scanning at 6×. |
| **`max`** | Hard ceiling (0.35) | Extreme | Critical cues only | Heavy dynamic compression + limiter | Ultra-speed maximum time contraction up to 10×. |
| **`uniform`** *(Control)* | None | None | None | None | Plain constant-rate stretch (what standard media players do). Used to compare effort against Speedman. |

---

## 🔒 Security & Path Allow-Lists

Speedman enforces strict path allow-listing. Only paths resolving inside the following workstation roots are accessible:
- `/mnt/d/Audio` (`D:\Audio`)
- `/mnt/d/Workspace` (`D:\Workspace`)
- `/mnt/d/Output` (`D:\Output`)
- `/mnt/d/AI` (`D:\AI`)
- `/mnt/d/OBVLT` (`D:\OBVLT`)
- `/tmp`

Any attempt to read from system paths (e.g. `/etc`, `C:\Windows`) returns `403 Forbidden`.
Both Windows (`D:\...`) and WSL (`/mnt/d/...`) formats are accepted and normalized automatically.
