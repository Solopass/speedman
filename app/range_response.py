"""HTTP 206 Partial Content range request helper for audio streaming."""
from __future__ import annotations

from pathlib import Path
from typing import Generator
from fastapi import HTTPException, Request
from starlette.responses import StreamingResponse


def range_stream_file(
    path: Path | str,
    request: Request,
    media_type: str = "audio/wav",
    chunk_size: int = 1024 * 64,
) -> StreamingResponse:
    path = Path(path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found")

    file_size = path.stat().st_size
    range_header = request.headers.get("range")

    if not range_header:
        def full_iter() -> Generator[bytes, None, None]:
            with open(path, "rb") as f:
                while chunk := f.read(chunk_size):
                    yield chunk

        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
        }
        return StreamingResponse(full_iter(), media_type=media_type, headers=headers)

    try:
        unit, spec = range_header.strip().split("=", 1)
        if unit != "bytes":
            raise ValueError()
        parts = spec.split("-", 1)
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else file_size - 1
    except Exception:
        raise HTTPException(
            status_code=416,
            detail="Invalid Range header",
            headers={"Content-Range": f"bytes */{file_size}"},
        )

    if start >= file_size or end >= file_size or start > end:
        raise HTTPException(
            status_code=416,
            detail="Requested range not satisfiable",
            headers={"Content-Range": f"bytes */{file_size}"},
        )

    content_length = end - start + 1

    def range_iter() -> Generator[bytes, None, None]:
        with open(path, "rb") as f:
            f.seek(start)
            remaining = content_length
            while remaining > 0:
                read_bytes = min(chunk_size, remaining)
                chunk = f.read(read_bytes)
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
    }
    return StreamingResponse(
        range_iter(),
        status_code=206,
        media_type=media_type,
        headers=headers,
    )
