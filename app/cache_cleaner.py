"""Cache statistics and cleanup utilities for Speedman.

Safely calculates disk usage and purges stale temporary downloads,
video caches, and blind comparison sets.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.paths import is_within

logger = logging.getLogger("speedman_cache")

CACHE_ROOTS = [
    Path("/mnt/d/Audio/Speed/downloads"),
    Path("/mnt/d/Audio/Speed/video-cache"),
    Path("/mnt/d/Audio/Speed/comparisons"),
]


def _get_target_dirs() -> List[Path]:
    seen = set()
    targets = []
    for d in CACHE_ROOTS:
        try:
            r = d.resolve()
            if r.is_dir() and str(r) not in seen:
                seen.add(str(r))
                targets.append(r)
        except Exception:
            pass
    return targets


def get_cache_stats() -> Dict[str, Any]:
    """Calculate total size and file count across temporary cache directories."""
    target_dirs = _get_target_dirs()
    total_bytes = 0
    total_files = 0
    dir_stats = []

    for d in target_dirs:
        dir_bytes = 0
        dir_files = 0
        try:
            for f in d.rglob("*"):
                if f.is_file() and not f.name.startswith("."):
                    try:
                        sz = f.stat().st_size
                        dir_bytes += sz
                        dir_files += 1
                    except OSError:
                        pass
        except Exception as e:
            logger.debug(f"[cache] Error scanning {d}: {e}")

        total_bytes += dir_bytes
        total_files += dir_files
        dir_stats.append({
            "path": str(d),
            "name": d.name,
            "bytes": dir_bytes,
            "mb": round(dir_bytes / (1024 * 1024), 2),
            "files": dir_files,
        })

    total_mb = round(total_bytes / (1024 * 1024), 2)
    total_gb = round(total_bytes / (1024 * 1024 * 1024), 2)

    return {
        "status": "success",
        "total_bytes": total_bytes,
        "total_mb": total_mb,
        "total_gb": total_gb,
        "total_files": total_files,
        "directories": dir_stats,
    }


def clean_cache(max_age_days: float = 7.0, all_files: bool = False) -> Dict[str, Any]:
    """Purge temporary cache files older than max_age_days, or all if all_files=True."""
    target_dirs = _get_target_dirs()
    now = time.time()
    max_age_s = float(max_age_days) * 86400.0

    stale_files: List[Tuple[Path, int]] = []
    for d in target_dirs:
        try:
            for f in d.rglob("*"):
                if f.is_file() and not f.name.startswith("."):
                    try:
                        mtime = f.stat().st_mtime
                        size = f.stat().st_size
                        age_s = now - mtime
                        if all_files or (age_s >= max_age_s):
                            stale_files.append((f, size))
                    except OSError:
                        pass
        except Exception as e:
            logger.warning(f"[cache] Error traversing {d}: {e}")

    deleted_count = 0
    reclaimed_bytes = 0
    for f, size in stale_files:
        try:
            f.unlink()
            deleted_count += 1
            reclaimed_bytes += size
        except Exception as e:
            logger.warning(f"[cache] Could not delete {f}: {e}")

    rec_mb = round(reclaimed_bytes / (1024 * 1024), 2)
    rec_gb = round(reclaimed_bytes / (1024 * 1024 * 1024), 2)

    return {
        "status": "success",
        "deleted_count": deleted_count,
        "reclaimed_bytes": reclaimed_bytes,
        "reclaimed_mb": rec_mb,
        "reclaimed_gb": rec_gb,
        "criteria": {
            "all_files": all_files,
            "max_age_days": max_age_days,
        },
    }
