"""Temporary video cache: watch a downloaded video, then keep it or let it go.

The default is to discard. Audio and transcripts are never touched -- they are the
point of Speedman; the video is scratch that exists so you can watch the thing once and
decide.

SAFETY. This module deletes files, so it is deliberately paranoid and deliberately small:

  * Nothing outside VIDEO_CACHE_DIR is ever deleted. Every path is re-checked with
    `is_within` immediately before unlinking, not merely when it was recorded.
  * Nothing outside the cache is ever touched, which is why the downloader moves (or,
    for a file that was already on disk, copies) the video in before anything is
    registered. Everything in the cache is Speedman's own copy, so discarding is always
    safe -- the question of ownership is settled before the file gets here, not by
    bookkeeping at delete time.
  * `pre_existed` records that D:\\Output\\Videos already held this video before we asked.
    Media API deduplicates by URL, so re-downloading last week's video returns the old
    file. That flag governs SAVING (the library already has it, so saving just drops our
    copy) -- it never makes a cache file undeletable.
  * A registry entry whose file has vanished is dropped, never chased.

Saving moves the file to MEDIA_LIBRARY_DIR, so anything left in the cache is by
definition disposable.
"""
from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

from app.paths import is_within, to_windows_path

logger = logging.getLogger("speedman_video")

OUTPUT_DIR = Path("/mnt/d/Audio/Speed")
VIDEO_CACHE_DIR = OUTPUT_DIR / "video-cache"
MEDIA_LIBRARY_DIR = Path("/mnt/d/Output/Videos")
REGISTRY_PATH = VIDEO_CACHE_DIR / "registry.json"

STALE_AFTER_S = 24 * 3600
"""Unsaved cache files older than this are swept at startup -- they are leftovers from a
session that never came back. Long enough that it cannot surprise anyone mid-decision."""

_lock = threading.Lock()


class VideoNotFound(RuntimeError):
    pass


class UnsafeDelete(RuntimeError):
    """Raised when a delete would leave the cache. Should never happen; if it does, the
    bookkeeping is wrong and stopping is better than guessing."""


@dataclass
class CachedVideo:
    video_id: str
    filename: str
    source_url: str
    created_at: float
    size_bytes: int
    pre_existed: bool
    """True when D:\\Output\\Videos already held this video before we asked. The cache
    copy is still ours to delete; this only means saving need not move anything."""
    saved: bool = False

    @property
    def path(self) -> Path:
        return VIDEO_CACHE_DIR / self.filename

    def public(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "filename": self.filename,
            "source_url": self.source_url,
            "size_mb": round(self.size_bytes / (1024 * 1024), 1),
            "saved": self.saved,
            "pre_existed": self.pre_existed,
            "stream_url": f"/api/v1/video/{self.video_id}/stream",
        }


# --------------------------------------------------------------------------- registry

def _load() -> dict[str, CachedVideo]:
    if not REGISTRY_PATH.is_file():
        return {}
    try:
        raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("[video] registry unreadable; starting empty")
        return {}
    out: dict[str, CachedVideo] = {}
    for entry in raw:
        try:
            out[entry["video_id"]] = CachedVideo(**entry)
        except Exception:
            continue
    return out


def _save(entries: dict[str, CachedVideo]) -> None:
    VIDEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(
        json.dumps([asdict(v) for v in entries.values()], indent=2), encoding="utf-8")


def register(path: Path, source_url: str, pre_existed: bool) -> CachedVideo:
    """Record a file already sitting in the cache directory."""
    path = Path(path)
    if not is_within(path, VIDEO_CACHE_DIR):
        raise UnsafeDelete(f"{path} is not inside the video cache")
    if not path.is_file():
        raise VideoNotFound(f"{path} does not exist")

    entry = CachedVideo(
        video_id=f"vid_{uuid.uuid4().hex[:10]}",
        filename=path.name,
        source_url=source_url,
        created_at=time.time(),
        size_bytes=path.stat().st_size,
        pre_existed=pre_existed,
    )
    with _lock:
        entries = _load()
        entries[entry.video_id] = entry
        _save(entries)
    logger.info(f"[video] cached {entry.filename} ({entry.public()['size_mb']} MB) as {entry.video_id}")
    return entry


def get(video_id: str) -> CachedVideo:
    entry = _load().get(str(video_id))
    if entry is None:
        raise VideoNotFound(f"no cached video {video_id!r}")
    return entry


def list_cached() -> list[dict[str, Any]]:
    return [v.public() for v in sorted(_load().values(), key=lambda v: v.created_at, reverse=True)]


def _forget(video_id: str) -> None:
    entries = _load()
    entries.pop(video_id, None)
    _save(entries)


# --------------------------------------------------------------------------- lifecycle

def save(video_id: str) -> dict[str, Any]:
    """Move the video into the media library, where the rest of the media lives."""
    with _lock:
        entries = _load()
        entry = entries.get(str(video_id))
        if entry is None:
            raise VideoNotFound(f"no cached video {video_id!r}")

        source = entry.path
        if not source.is_file():
            _forget(entry.video_id)
            raise VideoNotFound(f"{entry.filename} is no longer in the cache")

        # The library already has this one; our cache copy is redundant.
        if entry.pre_existed:
            existing = MEDIA_LIBRARY_DIR / entry.filename
            if existing.is_file():
                if is_within(source, VIDEO_CACHE_DIR):
                    source.unlink()
                _forget(entry.video_id)
                logger.info(f"[video] {entry.filename} already in the library; dropped cache copy")
                return {
                    "status": "already_saved",
                    "video_id": entry.video_id,
                    "path": str(existing),
                    "windows_path": to_windows_path(existing),
                }

        MEDIA_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
        target = MEDIA_LIBRARY_DIR / entry.filename
        # Never clobber something already in the library.
        if target.exists():
            stem, suffix = target.stem, target.suffix
            target = MEDIA_LIBRARY_DIR / f"{stem}_{int(time.time())}{suffix}"

        shutil.move(str(source), str(target))
        entry.saved = True
        entries.pop(entry.video_id, None)
        _save(entries)

    logger.info(f"[video] saved {entry.filename} -> {target}")
    return {
        "status": "saved",
        "video_id": entry.video_id,
        "path": str(target),
        "windows_path": to_windows_path(target),
    }


def discard(video_id: str) -> dict[str, Any]:
    """Delete the cached video -- the default when a viewer is closed.

    Always safe: the cache holds only Speedman's own copies. When the library already had
    the video, the original there is untouched and only our copy goes.
    """
    with _lock:
        entries = _load()
        entry = entries.get(str(video_id))
        if entry is None:
            raise VideoNotFound(f"no cached video {video_id!r}")

        path = entry.path
        if not path.is_file():
            _forget(entry.video_id)
            return {"status": "already_gone", "video_id": entry.video_id, "deleted": False}

        # Re-check containment at the moment of deletion, not just at registration.
        if not is_within(path, VIDEO_CACHE_DIR):
            raise UnsafeDelete(f"refusing to delete {path}: outside {VIDEO_CACHE_DIR}")

        path.unlink()
        _forget(entry.video_id)

    logger.info(f"[video] discarded {entry.filename}")
    return {
        "status": "discarded",
        "video_id": entry.video_id,
        "deleted": True,
        # Say plainly that the library copy survived, so the UI can too.
        "library_copy_kept": entry.pre_existed,
    }


def sweep(stale_after_s: float = STALE_AFTER_S) -> dict[str, Any]:
    """Drop registry entries whose files are gone, and discard unsaved leftovers older
    than `stale_after_s`. Called at startup so the cache cannot grow without bound."""
    removed, forgotten = [], []
    with _lock:
        entries = _load()
        now = time.time()
        for video_id, entry in list(entries.items()):
            if not entry.path.is_file():
                entries.pop(video_id, None)
                forgotten.append(entry.filename)
                continue
            if now - entry.created_at > stale_after_s:
                if is_within(entry.path, VIDEO_CACHE_DIR):
                    entry.path.unlink()
                    entries.pop(video_id, None)
                    removed.append(entry.filename)
        _save(entries)

    if removed or forgotten:
        logger.info(f"[video] sweep removed {len(removed)} stale, forgot {len(forgotten)} missing")
    return {"deleted": removed, "forgotten": forgotten}
