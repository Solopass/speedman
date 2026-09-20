r"""Windows <-> WSL path translation, shared by the API, the job queue and the Media API client.

Speedman runs under WSL but is driven from Windows (Explorer Send-To, the Studio UI,
desktop launchers), so every path crossing that boundary goes through here. Keeping one
copy avoids the three slightly-different conversions this module replaced.
"""
from __future__ import annotations

import re
from pathlib import Path

_WIN_DRIVE_RE = re.compile(r"^[a-zA-Z]:[\\/]")


def _strip_quotes(path_input: str | Path) -> str:
    return str(path_input).strip().strip('"').strip("'")


def normalize_path(path_input: str | Path) -> Path:
    r"""Seamlessly normalize Windows (D:\...) and WSL (/mnt/d/...) file paths."""
    s = _strip_quotes(path_input)
    if _WIN_DRIVE_RE.match(s):
        drive = s[0].lower()
        rest = s[2:].replace("\\", "/").lstrip("/")
        return Path(f"/mnt/{drive}/{rest}")
    return Path(s)


def to_windows_path(path_input: str | Path) -> str:
    r"""Convert WSL (/mnt/d/...) path to Windows (D:\...) path.

    Anything that is not under /mnt/<drive>/ is returned unchanged -- there is no
    Windows equivalent of, say, /tmp, and inventing one would be worse than echoing it.
    """
    s = _strip_quotes(path_input)
    if s.startswith("/mnt/") and len(s) > 6 and s[6] == "/":
        drive = s[5].upper()
        rest = s[7:].replace("/", "\\")
        return f"{drive}:\\{rest}"
    return s


def is_within(target: Path, root: Path) -> bool:
    """True when `target` is `root` or sits underneath it, after resolving symlinks."""
    try:
        resolved = target.resolve()
        resolved_root = root.resolve()
        return resolved == resolved_root or resolved.is_relative_to(resolved_root)
    except Exception:
        return False
