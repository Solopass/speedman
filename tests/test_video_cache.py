"""Tests for the temporary video cache.

This module deletes files, so the tests lean hard on what it must refuse to do. The
hazard is concrete: D:\\Output\\Videos already held a copy of the first video this feature
was tested against, downloaded days earlier.
"""
import json
import time

import pytest

from app import video as video_cache
from app.video import CachedVideo, VideoNotFound, UnsafeDelete


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """Point the whole module at a throwaway tree."""
    cache_dir = tmp_path / "video-cache"
    library = tmp_path / "library"
    cache_dir.mkdir()
    library.mkdir()
    monkeypatch.setattr(video_cache, "VIDEO_CACHE_DIR", cache_dir)
    monkeypatch.setattr(video_cache, "MEDIA_LIBRARY_DIR", library)
    monkeypatch.setattr(video_cache, "REGISTRY_PATH", cache_dir / "registry.json")
    return cache_dir, library


def make_file(directory, name="clip.mp4", content=b"video-bytes"):
    p = directory / name
    p.write_bytes(content)
    return p


# --------------------------------------------------------------------------- registration

def test_registers_a_file_inside_the_cache(cache):
    cache_dir, _ = cache
    entry = video_cache.register(make_file(cache_dir), "https://x/v", pre_existed=False)
    assert entry.video_id.startswith("vid_")
    assert video_cache.get(entry.video_id).filename == "clip.mp4"
    assert entry.public()["stream_url"].endswith("/stream")


def test_refuses_to_register_a_file_outside_the_cache(cache, tmp_path):
    """Registering an outside path would make it a deletion target."""
    outside = make_file(tmp_path, "elsewhere.mp4")
    with pytest.raises(UnsafeDelete):
        video_cache.register(outside, "https://x/v", pre_existed=False)


def test_register_requires_the_file_to_exist(cache):
    cache_dir, _ = cache
    with pytest.raises(VideoNotFound):
        video_cache.register(cache_dir / "ghost.mp4", "https://x/v", pre_existed=False)


# --------------------------------------------------------------------------- discard

def test_discard_deletes_only_the_cache_copy(cache):
    cache_dir, _ = cache
    path = make_file(cache_dir)
    entry = video_cache.register(path, "https://x/v", pre_existed=False)

    result = video_cache.discard(entry.video_id)
    assert result["deleted"] is True
    assert not path.exists()
    with pytest.raises(VideoNotFound):
        video_cache.get(entry.video_id)


def test_discard_never_touches_the_library_original(cache):
    """The library copy predates us and must survive discarding our working copy."""
    cache_dir, library = cache
    original = make_file(library, "clip.mp4", b"the-users-original")
    copy = make_file(cache_dir, "clip.mp4", b"our-copy")
    entry = video_cache.register(copy, "https://x/v", pre_existed=True)

    result = video_cache.discard(entry.video_id)
    assert result["deleted"] is True
    assert result["library_copy_kept"] is True
    assert not copy.exists()
    assert original.exists()
    assert original.read_bytes() == b"the-users-original"


def test_discard_of_a_vanished_file_is_not_an_error(cache):
    cache_dir, _ = cache
    path = make_file(cache_dir)
    entry = video_cache.register(path, "https://x/v", pre_existed=False)
    path.unlink()
    assert video_cache.discard(entry.video_id)["status"] == "already_gone"


def test_discard_refuses_a_path_that_escaped_the_cache(cache, tmp_path, monkeypatch):
    """Containment is re-checked at deletion, not merely at registration -- a registry
    edited by hand, or a moved cache dir, must not become a delete anywhere primitive."""
    cache_dir, _ = cache
    path = make_file(cache_dir)
    entry = video_cache.register(path, "https://x/v", pre_existed=False)

    # Repoint the cache so the recorded file now resolves outside it.
    elsewhere = tmp_path / "moved-cache"
    elsewhere.mkdir()
    monkeypatch.setattr(video_cache, "VIDEO_CACHE_DIR", elsewhere)
    (elsewhere / "registry.json").write_text(
        json.dumps([{**entry.__dict__}]), encoding="utf-8")
    monkeypatch.setattr(video_cache, "REGISTRY_PATH", elsewhere / "registry.json")

    # The recorded filename now points at a file that does not exist in the new cache,
    # so this resolves as already_gone rather than deleting the old one.
    result = video_cache.discard(entry.video_id)
    assert result["deleted"] is False
    assert path.exists(), "the original file outside the new cache must survive"


def test_discard_unknown_id_is_an_error(cache):
    with pytest.raises(VideoNotFound):
        video_cache.discard("vid_nope")


# --------------------------------------------------------------------------- save

def test_save_moves_the_video_into_the_library(cache):
    cache_dir, library = cache
    path = make_file(cache_dir)
    entry = video_cache.register(path, "https://x/v", pre_existed=False)

    result = video_cache.save(entry.video_id)
    assert result["status"] == "saved"
    assert not path.exists()
    assert (library / "clip.mp4").read_bytes() == b"video-bytes"
    assert result["windows_path"].startswith("/") or ":" in result["windows_path"]


def test_save_never_clobbers_an_existing_library_file(cache):
    cache_dir, library = cache
    existing = make_file(library, "clip.mp4", b"already-here")
    path = make_file(cache_dir, "clip.mp4", b"newly-downloaded")
    entry = video_cache.register(path, "https://x/v", pre_existed=False)

    result = video_cache.save(entry.video_id)
    assert existing.read_bytes() == b"already-here"
    assert result["path"] != str(existing)
    assert len(list(library.glob("clip*.mp4"))) == 2


def test_saving_something_the_library_already_has_just_drops_the_copy(cache):
    cache_dir, library = cache
    original = make_file(library, "clip.mp4", b"original")
    copy = make_file(cache_dir, "clip.mp4", b"copy")
    entry = video_cache.register(copy, "https://x/v", pre_existed=True)

    result = video_cache.save(entry.video_id)
    assert result["status"] == "already_saved"
    assert not copy.exists()
    assert original.read_bytes() == b"original"


def test_save_of_a_vanished_file_reports_clearly(cache):
    cache_dir, _ = cache
    path = make_file(cache_dir)
    entry = video_cache.register(path, "https://x/v", pre_existed=False)
    path.unlink()
    with pytest.raises(VideoNotFound):
        video_cache.save(entry.video_id)


# --------------------------------------------------------------------------- sweep

def test_sweep_removes_stale_entries_but_spares_fresh_ones(cache):
    cache_dir, _ = cache
    old = make_file(cache_dir, "old.mp4")
    fresh = make_file(cache_dir, "fresh.mp4")
    old_entry = video_cache.register(old, "https://x/old", pre_existed=False)
    video_cache.register(fresh, "https://x/fresh", pre_existed=False)

    # Age the first entry past the threshold.
    entries = video_cache._load()
    entries[old_entry.video_id].created_at = time.time() - 48 * 3600
    video_cache._save(entries)

    result = video_cache.sweep()
    assert "old.mp4" in result["deleted"]
    assert not old.exists()
    assert fresh.exists()


def test_sweep_forgets_entries_whose_files_are_gone(cache):
    cache_dir, _ = cache
    path = make_file(cache_dir)
    entry = video_cache.register(path, "https://x/v", pre_existed=False)
    path.unlink()

    assert "clip.mp4" in video_cache.sweep()["forgotten"]
    with pytest.raises(VideoNotFound):
        video_cache.get(entry.video_id)


def test_sweep_on_an_empty_cache_is_harmless(cache):
    assert video_cache.sweep() == {"deleted": [], "forgotten": []}


def test_corrupt_registry_does_not_crash(cache):
    cache_dir, _ = cache
    (cache_dir / "registry.json").write_text("{ not json", encoding="utf-8")
    assert video_cache.list_cached() == []
