"""Tests for the eval harness's WER implementation.

The harness is only as trustworthy as its metric, and every tuning decision will be
made from these numbers -- so the metric gets tested like production code.
"""
import pytest

from eval.wer import normalise, tokens, wer, cer


def test_normalise_strips_punctuation_and_case():
    assert normalise("Hello, World!") == "hello world"
    assert normalise("  multiple   spaces  ") == "multiple spaces"


def test_normalise_keeps_internal_apostrophes():
    """parakeet emits contractions; dropping the apostrophe would split one word
    into two and inflate WER for no acoustic reason."""
    assert normalise("y'all don't") == "y'all don't"
    assert normalise("'quoted' word") == "quoted word"


def test_normalise_unifies_curly_apostrophes():
    assert normalise("don’t") == normalise("don't")


def test_identical_text_is_zero_error():
    r = wer("the quick brown fox", "the quick brown fox")
    assert r.rate == 0.0
    assert r.errors == 0


def test_punctuation_and_case_differences_are_not_errors():
    assert wer("Hello, world.", "hello world").rate == 0.0


def test_single_substitution():
    r = wer("the quick brown fox", "the quick red fox")
    assert r.errors == 1
    assert r.substitutions == 1
    assert r.rate == pytest.approx(0.25)


def test_single_deletion():
    r = wer("the quick brown fox", "the quick fox")
    assert r.errors == 1
    assert r.deletions == 1


def test_single_insertion():
    r = wer("the quick fox", "the quick brown fox")
    assert r.errors == 1
    assert r.insertions == 1


def test_mixed_edits_count_correctly():
    r = wer("a b c d e", "a x c e f")
    # b->x substitution, d deleted, f inserted
    assert r.errors == 3
    assert r.length == 5
    assert r.rate == pytest.approx(0.6)


def test_empty_hypothesis_is_total_loss():
    r = wer("the quick brown fox", "")
    assert r.rate == pytest.approx(1.0)
    assert r.deletions == 4


def test_empty_reference_reports_insertions_without_dividing_by_zero():
    r = wer("", "spurious words here")
    assert r.length == 0
    assert r.rate == 0.0
    assert r.insertions == 3


def test_rate_can_exceed_one_when_hypothesis_is_longer():
    """A speed-mangled clip can make ASR hallucinate more words than the reference."""
    r = wer("one two", "a b c d e f")
    assert r.rate > 1.0


def test_cer_is_gentler_than_wer_on_near_misses():
    """A one-character slip destroys a whole word for WER but barely moves CER --
    which is why both are reported."""
    w = wer("intelligible speech", "inteligible speech")
    c = cer("intelligible speech", "inteligible speech")
    assert w.rate == pytest.approx(0.5)
    assert c.rate < 0.1


def test_tokens_handles_empty_and_whitespace():
    assert tokens("") == []
    assert tokens("   ") == []
    assert tokens("one") == ["one"]


# --------------------------------------------------------------------------- sweep statistics

from eval.sweep import _sign_test_p, _paired


def test_sign_test_symmetric_split_is_not_significant():
    assert _sign_test_p(6, 12) == pytest.approx(1.0)


def test_sign_test_unanimous_result_is_significant():
    assert _sign_test_p(12, 12) < 0.001
    assert _sign_test_p(0, 12) < 0.001


def test_sign_test_matches_known_binomial_values():
    # 1 win in 12 two-sided: 2 * (C(12,0) + C(12,1)) / 2^12 = 26/4096
    assert _sign_test_p(1, 12) == pytest.approx(26 / 4096)
    assert _sign_test_p(11, 12) == pytest.approx(26 / 4096)


def test_sign_test_is_symmetric_in_wins_and_losses():
    for wins in range(13):
        assert _sign_test_p(wins, 12) == pytest.approx(_sign_test_p(12 - wins, 12))


def test_sign_test_handles_empty_sample():
    assert _sign_test_p(0, 0) == 1.0


def test_paired_counts_wins_as_lower_wer():
    base = {"a": 0.5, "b": 0.5, "c": 0.5}
    better = {"a": 0.4, "b": 0.4, "c": 0.4}
    assert _paired(better, base).startswith("3/3")
    worse = {"a": 0.6, "b": 0.6, "c": 0.6}
    assert _paired(worse, base).startswith("0/3")


def test_paired_excludes_ties_from_the_sample():
    base = {"a": 0.5, "b": 0.5, "c": 0.5}
    mixed = {"a": 0.4, "b": 0.5, "c": 0.6}  # one win, one tie, one loss
    assert _paired(mixed, base).startswith("1/2")


def test_paired_marks_only_significant_improvements():
    base = {f"c{i}": 0.5 for i in range(12)}
    strong = {f"c{i}": 0.4 for i in range(12)}
    assert "✓" in _paired(strong, base)

    # A significant result in the WRONG direction must not be marked as a win.
    regression = {f"c{i}": 0.6 for i in range(12)}
    assert "✓" not in _paired(regression, base)


def test_paired_handles_no_shared_clips():
    assert _paired({"x": 0.1}, {"y": 0.2}) == "--"
