"""Tests for the listening-test harness.

The blinding tests are the important ones. A listening test whose blind leaks is worse
than no listening test: it produces confident numbers that measure expectation.
"""
import json

import pytest

from eval import listening
from eval.listening import (
    Session, Trial, Verdict, InvalidVerdict, SessionNotFound,
    record_verdict, load_session, reveal, summarise,
)


def make_trial(index=0, speed=5.0, slot_a="speedman", slot_b="uniform"):
    return Trial(index=index, speed=speed, clip_label="c", source="/x.mp3",
                 offset_s=0.0, duration_s=30.0, slot_a=slot_a, slot_b=slot_b)


def make_session(trials=None, session_id="lt_test"):
    return Session(session_id=session_id, created_at=0.0,
                   trials=trials if trials is not None else [make_trial()])


@pytest.fixture
def isolated_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(listening, "LISTENING_DIR", tmp_path)
    return tmp_path


# --------------------------------------------------------------------------- blinding

def test_blind_trial_payload_never_names_a_condition():
    trial = make_trial()
    assert "speedman" not in json.dumps(trial.blind())
    assert "uniform" not in json.dumps(trial.blind())


def test_blind_session_state_never_names_a_condition():
    session = make_session([make_trial(0), make_trial(1, slot_a="uniform", slot_b="speedman")])
    record = json.dumps(session.blind_state())
    assert "speedman" not in record
    assert "uniform" not in record


def test_blind_state_still_carries_verdicts_for_resume(isolated_dir):
    session = make_session([make_trial(0), make_trial(1)])
    session.directory.mkdir(parents=True, exist_ok=True)
    record_verdict(session, 0, "a", "both")
    state = session.blind_state()
    assert state["completed"] == 1
    assert state["next_index"] == 1
    assert state["verdicts"][0]["choice"] == "a"


def test_reveal_refuses_until_every_trial_is_answered(isolated_dir):
    session = make_session([make_trial(0), make_trial(1)])
    session.directory.mkdir(parents=True, exist_ok=True)
    record_verdict(session, 0, "a", "both")
    with pytest.raises(InvalidVerdict, match="finish it"):
        reveal(session)


def test_reveal_unblinds_once_complete(isolated_dir):
    session = make_session([make_trial(0)])
    session.directory.mkdir(parents=True, exist_ok=True)
    record_verdict(session, 0, "a", "both")
    result = reveal(session)
    assert result["trials"][0]["slot_a"] == "speedman"
    assert result["trials"][0]["winner"] == "speedman"


# --------------------------------------------------------------------------- verdicts

def test_winner_maps_choice_through_the_randomised_slots():
    """The point of randomising order: 'a' means Speedman in one trial and uniform in
    the next, so the mapping must be applied per trial."""
    normal = make_trial(0, slot_a="speedman", slot_b="uniform")
    flipped = make_trial(1, slot_a="uniform", slot_b="speedman")
    assert Verdict(0, "a", "both").winner(normal) == "speedman"
    assert Verdict(1, "a", "both").winner(flipped) == "uniform"
    assert Verdict(1, "b", "both").winner(flipped) == "speedman"


def test_no_difference_has_no_winner():
    assert Verdict(0, "none", "both").winner(make_trial()) is None


@pytest.mark.parametrize("choice", ["x", "invalid", "", "left"])
def test_invalid_choice_is_rejected(isolated_dir, choice):
    session = make_session()
    session.directory.mkdir(parents=True, exist_ok=True)
    with pytest.raises(InvalidVerdict):
        record_verdict(session, 0, choice, "both")


def test_invalid_followed_is_rejected(isolated_dir):
    session = make_session()
    session.directory.mkdir(parents=True, exist_ok=True)
    with pytest.raises(InvalidVerdict):
        record_verdict(session, 0, "a", "sort of")


def test_picked_is_incoherent_with_no_difference(isolated_dir):
    """If you could not tell them apart, 'I followed the one I picked' says nothing."""
    session = make_session()
    session.directory.mkdir(parents=True, exist_ok=True)
    with pytest.raises(InvalidVerdict, match="meaningless"):
        record_verdict(session, 0, "none", "picked")


def test_verdict_for_unknown_trial_is_rejected(isolated_dir):
    session = make_session()
    session.directory.mkdir(parents=True, exist_ok=True)
    with pytest.raises(InvalidVerdict):
        record_verdict(session, 99, "a", "both")


def test_choice_and_followed_are_normalised(isolated_dir):
    session = make_session()
    session.directory.mkdir(parents=True, exist_ok=True)
    v = record_verdict(session, 0, " A ", "BOTH")
    assert v.choice == "a" and v.followed == "both"


# --------------------------------------------------------------------------- persistence

def test_verdicts_survive_a_reload(isolated_dir):
    """Resuming after a browser crash must not lose work."""
    session = make_session([make_trial(0), make_trial(1)], session_id="lt_resume")
    session.directory.mkdir(parents=True, exist_ok=True)
    listening._save_manifest(session)
    record_verdict(session, 0, "b", "picked")

    reloaded = load_session("lt_resume")
    assert len(reloaded.verdicts) == 1
    assert reloaded.verdicts[0].choice == "b"
    assert reloaded.next_index() == 1
    assert reloaded.trials[0].slot_a == "speedman"


def test_load_session_rejects_a_missing_id(isolated_dir):
    with pytest.raises(SessionNotFound):
        load_session("does_not_exist")


def test_session_id_cannot_escape_the_listening_directory(isolated_dir):
    """A crafted id must not read manifest.json from elsewhere on disk."""
    with pytest.raises(SessionNotFound):
        load_session("../../../etc")


def test_audio_path_rejects_an_unknown_slot():
    with pytest.raises(InvalidVerdict):
        make_session().audio_path(0, "c")


# --------------------------------------------------------------------------- summary

def test_summarise_counts_wins_ties_and_followability(isolated_dir):
    trials = [
        make_trial(0, speed=5.0, slot_a="speedman", slot_b="uniform"),
        make_trial(1, speed=5.0, slot_a="uniform", slot_b="speedman"),
        make_trial(2, speed=5.0),
        make_trial(3, speed=8.0),
    ]
    session = make_session(trials)
    session.directory.mkdir(parents=True, exist_ok=True)
    record_verdict(session, 0, "a", "both")       # speedman
    record_verdict(session, 1, "a", "both")       # uniform
    record_verdict(session, 2, "none", "neither")  # tie
    record_verdict(session, 3, "b", "picked")      # uniform at 8x

    at5 = next(r for r in summarise(session) if r["speed"] == 5.0)
    assert (at5["speedman"], at5["uniform"], at5["ties"]) == (1, 1, 1)
    assert at5["n_decisive"] == 2 and at5["n_trials"] == 3
    assert at5["followed_both"] == 2 and at5["followed_neither"] == 1

    at8 = next(r for r in summarise(session) if r["speed"] == 8.0)
    assert at8["uniform"] == 1 and at8["followed_picked"] == 1


def test_summarise_ignores_unanswered_trials(isolated_dir):
    session = make_session([make_trial(0), make_trial(1)])
    session.directory.mkdir(parents=True, exist_ok=True)
    record_verdict(session, 0, "a", "both")
    assert summarise(session)[0]["n_trials"] == 1


# --------------------------------------------------------------------------- build guards

def test_build_session_rejects_an_empty_speed_list():
    with pytest.raises(ValueError, match="at least one speed"):
        listening.build_session(speeds=[])


def test_build_session_rejects_zero_trials():
    with pytest.raises(ValueError, match="at least one trial"):
        listening.build_session(speeds=[5.0], trials_per_speed=0)


def test_build_session_reports_when_no_clips_are_available():
    with pytest.raises(ValueError, match="no clips available"):
        listening.build_session(speeds=[5.0], clips=[])


# --------------------------------------------------------------------------- API surface

from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)


def test_clips_endpoint_lists_the_eval_manifest():
    resp = client.get("/api/v1/listening/clips")
    assert resp.status_code == 200
    labels = {c["label"] for c in resp.json()}
    assert "podcast_dating" in labels


def test_unknown_session_is_404():
    assert client.get("/api/v1/listening/session/nope").status_code == 404
    assert client.post("/api/v1/listening/session/nope/verdict",
                       json={"trial_index": 0, "choice": "a", "followed": "both"}).status_code == 404


def test_audio_endpoint_rejects_a_slot_that_is_not_a_or_b(isolated_dir):
    session = make_session([make_trial(0)], session_id="lt_api")
    session.directory.mkdir(parents=True, exist_ok=True)
    listening._save_manifest(session)
    assert client.get(f"/api/v1/listening/lt_api/audio/0/c").status_code == 400


def test_session_creation_rejects_unknown_clip_labels():
    resp = client.post("/api/v1/listening/session", json={
        "speeds": [5.0], "trials_per_speed": 1, "clips": ["not_a_clip"],
    })
    assert resp.status_code == 400
    assert "not_a_clip" in resp.json()["detail"]


def test_session_creation_validates_its_inputs():
    assert client.post("/api/v1/listening/session",
                       json={"speeds": [5.0], "trials_per_speed": 0}).status_code == 422
    assert client.post("/api/v1/listening/session",
                       json={"speeds": [5.0], "excerpt_s": 1.0}).status_code == 422


def test_results_endpoint_refuses_to_unblind_an_unfinished_session(isolated_dir):
    """409 rather than 200-with-nothing: peeking mid-session would bias the rest."""
    session = make_session([make_trial(0), make_trial(1)], session_id="lt_partial")
    session.directory.mkdir(parents=True, exist_ok=True)
    listening._save_manifest(session)
    record_verdict(session, 0, "a", "both")
    resp = client.get("/api/v1/listening/session/lt_partial/results")
    assert resp.status_code == 409
