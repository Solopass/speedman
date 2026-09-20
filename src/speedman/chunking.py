"""Long-audio chunking engine for Speedman.

Cuts multi-hour audiobooks and podcasts on natural boundary pauses identified by VAD,
processes each block independently, and stitches them with a micro-fade so there are no
phase clicks at the seams. Chunking is what makes per-chunk progress reporting and
mid-run cancellation possible, and it bounds the size of the arrays the *backend* sees.

It does NOT bound total process memory: `io.load` decodes the whole input up front, and
`process_chunked` accumulates every output chunk before concatenating, so peak usage is
roughly input + output + one output copy. A 10-hour file at 24 kHz float32 is ~3.5 GB of
input on its own. Streaming chunks to disk and concatenating at the file level is the fix
if that ceiling ever matters.
"""
from __future__ import annotations

import time
from typing import Callable, Optional, List, Tuple
import numpy as np

from . import analyze, post, ratemap
from .config import Config
from .stretch import get_backend


class CancelledError(Exception):
    """Raised when an active processing job is aborted by the user."""
    pass


def find_silence_cutpoint(
    y_window: np.ndarray,
    sr: int,
    nominal_sample: int,
    window_start_sample: int,
) -> int:
    """Finds the quietest point (or silence run) in a search window to cut audio cleanly.
    
    Returns the absolute sample index in y.
    """
    if len(y_window) < sr:  # Less than 1 second window
        return nominal_sample

    hop = analyze.HOP
    frame_db = analyze._frame_db(y_window, sr, hop=hop, frame_ms=25.0)

    # Search for frames below silence threshold (-40 dB or minimum in window)
    min_db = np.min(frame_db)
    # Prefer frames close to nominal cutpoint that are near min_db
    target_frame = (nominal_sample - window_start_sample) // hop
    target_frame = max(0, min(len(frame_db) - 1, target_frame))

    # Candidate frames within 6 dB of the quietest moment in this window
    quiet_mask = frame_db <= max(min_db + 6.0, -35.0)
    quiet_indices = np.flatnonzero(quiet_mask)

    if len(quiet_indices) > 0:
        # Pick the quiet frame closest to the nominal target
        closest_idx = quiet_indices[np.argmin(np.abs(quiet_indices - target_frame))]
        best_sample = window_start_sample + (closest_idx * hop)
        return int(best_sample)

    # Fallback to absolute minimum energy frame
    best_frame = int(np.argmin(frame_db))
    return int(window_start_sample + (best_frame * hop))


def plan_chunks(
    n_samples: int,
    sr: int,
    y: Optional[np.ndarray] = None,
    target_chunk_min: float = 25.0,
    search_window_s: float = 60.0,
) -> List[Tuple[int, int]]:
    """Calculates [start, end) sample boundaries for all chunks.
    
    If y is provided, cutpoints are aligned to natural speech pauses.
    """
    total_dur_s = n_samples / sr
    chunk_dur_s = target_chunk_min * 60.0

    if total_dur_s <= chunk_dur_s * 1.2:
        return [(0, n_samples)]

    n_chunks = max(2, int(np.ceil(total_dur_s / chunk_dur_s)))
    nominal_interval = n_samples / n_chunks
    search_samples = int(search_window_s * sr)

    cutpoints = [0]
    for i in range(1, n_chunks):
        nominal_cut = int(i * nominal_interval)
        if y is not None:
            win_start = max(cutpoints[-1] + int(sr * 10), nominal_cut - search_samples)
            win_end = min(n_samples - int(sr * 10), nominal_cut + search_samples)
            if win_end > win_start:
                y_win = y[win_start:win_end]
                cut = find_silence_cutpoint(y_win, sr, nominal_cut, win_start)
            else:
                cut = nominal_cut
        else:
            cut = nominal_cut
        cutpoints.append(cut)

    cutpoints.append(n_samples)

    # Form (start, end) pairs
    chunks = []
    for s, e in zip(cutpoints[:-1], cutpoints[1:]):
        if e > s:
            chunks.append((s, e))
    return chunks


def process_chunked(
    y: np.ndarray,
    sr: int,
    cfg: Config,
    target_chunk_min: float = 25.0,
    on_progress: Optional[Callable[[dict], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Tuple[np.ndarray, dict, dict]:
    """Processes audio in sequential chunks with progress updates and cancellation checks.
    
    Returns (output_audio, aggregate_timings, aggregate_notes).
    """
    n_samples = len(y)
    chunks = plan_chunks(n_samples, sr, y=y, target_chunk_min=target_chunk_min)
    total_chunks = len(chunks)

    t_start = time.perf_counter()
    backend = get_backend(cfg.backend)
    out_pieces: List[np.ndarray] = []
    chunk_durations_in: List[float] = []
    chunk_durations_out: List[float] = []

    timings = {"analyze": 0.0, "stretch": 0.0, "post": 0.0, "total": 0.0}
    silence_fractions = []
    effective_rates = []
    estimated_rates = []
    clamped_fractions = []
    # Global time map, stitched as we go: each chunk's map is relative to its own start,
    # so both axes need the running offsets applied before they are concatenated.
    map_in_bounds: List[np.ndarray] = []
    map_out_pos: List[np.ndarray] = []
    map_rates: List[np.ndarray] = []
    out_offset = 0.0

    for idx, (c_start, c_end) in enumerate(chunks, start=1):
        if cancel_check and cancel_check():
            raise CancelledError("Processing cancelled by user")

        c_len = c_end - c_start
        c_dur = c_len / sr
        chunk_durations_in.append(c_dur)
        y_chunk = y[c_start:c_end]

        t_chunk_start = time.perf_counter()

        # Step 1: Analyze & Time Map
        t0 = time.perf_counter()
        if cfg.uniform:
            tm = None
            silence_fractions.append(0.0)
            effective_rates.append(cfg.speed)
            estimated_rates.append(cfg.speed)
            clamped_fractions.append(0.0)
        else:
            ann = analyze.annotate_vad(
                y_chunk, sr,
                protect_window_ms=cfg.ratemap.protect_window_ms,
                protect_lead_ms=cfg.ratemap.protect_lead_ms,
            )
            tm = ratemap.build_time_map(ann, cfg.speed, cfg.ratemap)
            s_frac = tm.notes.get("silence_fraction", 0.0)
            silence_fractions.append(s_frac)
            # Measured from the solved map; see ratemap.TimeMap.measured_speech_rate.
            effective_rates.append(tm.measured_speech_rate([sp.kind.value for sp in ann.spans]))
            estimated_rates.append(cfg.speed * (1.0 - 0.6 * s_frac))
            clamped_fractions.append(tm.clamped_fraction(cfg.ratemap.max_rate))
        timings["analyze"] += time.perf_counter() - t0

        if cancel_check and cancel_check():
            raise CancelledError("Processing cancelled by user")

        # Step 2: Rubberband Stretch
        t0 = time.perf_counter()
        if tm is None:
            c_out = backend.staged(y_chunk, sr, cfg.speed) if cfg.staged else backend.stretch(y_chunk, sr, cfg.speed)
        else:
            c_out = backend.stretch_map(y_chunk, sr, tm)
        timings["stretch"] += time.perf_counter() - t0

        if cancel_check and cancel_check():
            raise CancelledError("Processing cancelled by user")

        # Step 3: Post Clarity Chain
        t0 = time.perf_counter()
        c_out = post.run(c_out, sr, cfg.post)
        timings["post"] += time.perf_counter() - t0

        # Cosine fade edges slightly to guarantee 0 clicks across chunk boundaries
        if len(c_out) > sr // 10:
            fade_len = min(int(sr * 0.005), 128)  # 5ms micro-fade
            fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            c_out[:fade_len] *= fade_in
            c_out[-fade_len:] *= fade_in[::-1]

        # Stitch this chunk's map into the global one before moving on. The output axis
        # advances by the rendered length, not by the map's own total, so any drift
        # introduced by the post chain stays confined to one chunk instead of
        # accumulating across the file.
        if tm is not None:
            seg_out = tm.segment_output_lengths()
            map_in_bounds.append(c_start + tm.seg_bounds[:-1].astype(np.float64))
            map_out_pos.append(out_offset + np.concatenate([[0.0], np.cumsum(seg_out)[:-1]]))
            map_rates.append(tm.rates)
        out_offset += len(c_out)

        out_pieces.append(c_out)
        chunk_durations_out.append(len(c_out) / sr)

        # Progress reporting & Measured ETA
        elapsed = time.perf_counter() - t_start
        pct = round((idx / total_chunks) * 100, 1)
        # Measured ETA based on average elapsed time per completed chunk
        avg_chunk_time = elapsed / idx
        remaining_chunks = total_chunks - idx
        eta_seconds = round(avg_chunk_time * remaining_chunks, 1)

        if on_progress:
            on_progress({
                "stage": "stretching",
                "current_chunk": idx,
                "total_chunks": total_chunks,
                "progress_pct": pct,
                "elapsed_s": round(elapsed, 1),
                "eta_s": eta_seconds,
            })

    output_audio = np.concatenate(out_pieces) if out_pieces else np.zeros(0, dtype=np.float32)
    timings["total"] = time.perf_counter() - t_start

    # Weighted by input duration: chunks are pause-aligned and the last one is usually
    # short, so an unweighted mean would let a 30-second tail count as much as 25 minutes.
    weights = np.asarray(chunk_durations_in[:len(silence_fractions)], dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones(len(silence_fractions), dtype=np.float64)

    def weighted(values, default):
        vals = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(vals)
        if not finite.any():
            return float(default)
        return float(np.average(vals[finite], weights=weights[:len(vals)][finite]))

    mean_silence = weighted(silence_fractions, 0.0) if silence_fractions else 0.0
    mean_eff_rate = weighted(effective_rates, cfg.speed) if effective_rates else cfg.speed

    notes = {
        "chunked": True,
        "total_chunks": total_chunks,
        "silence_fraction": round(mean_silence, 4),
        "effective_speech_rate": round(mean_eff_rate, 2),
        "estimated_speech_rate": round(weighted(estimated_rates, cfg.speed), 2),
        "rate_clamped_fraction": round(weighted(clamped_fractions, 0.0), 4),
        "input_duration_s": round(n_samples / sr, 2),
        "output_duration_s": round(len(output_audio) / sr, 2),
    }

    stitched = _stitch_time_map(map_in_bounds, map_out_pos, map_rates,
                                n_samples, len(output_audio), sr)
    return output_audio, timings, notes, stitched


def _stitch_time_map(in_bounds, out_pos, rates, n_in: int, n_out: int, sr: int):
    """One TimeMap spanning the whole file, from the per-chunk maps.

    Returns None in uniform mode, where there is no map to stitch and t/N is exact.
    """
    from .ratemap import TimeMap

    if not in_bounds:
        return None

    bounds = np.concatenate(in_bounds + [np.array([float(n_in)])])
    positions = np.concatenate(out_pos + [np.array([float(n_out)])])
    all_rates = np.concatenate(rates)

    # np.interp needs a strictly increasing x, and chunk seams can land on the same
    # sample when a boundary pause is cut to nothing.
    keep = np.concatenate([[True], np.diff(bounds) > 0])
    bounds, positions = bounds[keep], positions[keep]
    positions = np.maximum.accumulate(positions)

    anchors = np.stack([np.rint(bounds).astype(np.int64),
                        np.rint(positions).astype(np.int64)], axis=1)
    return TimeMap(
        anchors=anchors,
        n_in=n_in,
        n_out=n_out,
        rates=all_rates,
        seg_bounds=np.rint(bounds).astype(np.int64),
        notes={"stitched_from_chunks": len(in_bounds)},
    )
