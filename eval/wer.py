"""Word error rate, with the text normalisation that makes it meaningful.

Deliberately dependency-free. `pyproject.toml` lists jiwer under the `eval` extra,
but WER is ~30 lines and pulling a dependency (and, via the same extra, torch) for
it would make the harness harder to run than the thing it measures.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Keep apostrophes inside words ("y'all", "don't") -- parakeet emits them and
# dropping them would inflate WER on contractions for no reason.
_PUNCT = re.compile(r"[^\w\s']|(?<!\w)'|'(?!\w)")
_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Applied identically to reference and hypothesis, so it can only remove
    differences that are not intelligibility differences.
    """
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def tokens(text: str) -> list[str]:
    n = normalise(text)
    return n.split() if n else []


@dataclass(frozen=True)
class ErrorRate:
    errors: int
    length: int
    substitutions: int
    deletions: int
    insertions: int

    @property
    def rate(self) -> float:
        """Errors per reference token. Can exceed 1.0 when the hypothesis is longer."""
        return self.errors / self.length if self.length else 0.0

    def __str__(self) -> str:
        return f"{self.rate:.4f} ({self.errors}/{self.length})"


def _levenshtein(ref: list, hyp: list) -> ErrorRate:
    """Edit distance with operation counts, O(len(ref) * len(hyp)) time, O(len(hyp)) space.

    Two rolling rows: a 90s clip is ~250 words, so the full matrix would be fine, but
    the whole grid runs this a few hundred times and rows keep it trivially cheap.
    """
    n, m = len(ref), len(hyp)
    if n == 0:
        return ErrorRate(m, 0, 0, 0, m)

    # Each cell carries (cost, subs, dels, ins) so we can report the breakdown.
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)] + [None] * m
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                c, s, d, ins = prev[j - 1]
                cur[j] = (c, s, d, ins)
                continue
            sub = prev[j - 1]
            dele = prev[j]
            insert = cur[j - 1]
            best = min(sub[0], dele[0], insert[0]) + 1
            if sub[0] <= dele[0] and sub[0] <= insert[0]:
                cur[j] = (best, sub[1] + 1, sub[2], sub[3])
            elif dele[0] <= insert[0]:
                cur[j] = (best, dele[1], dele[2] + 1, dele[3])
            else:
                cur[j] = (best, insert[1], insert[2], insert[3] + 1)
        prev = cur

    cost, subs, dels, ins = prev[m]
    return ErrorRate(cost, n, subs, dels, ins)


def wer(reference: str, hypothesis: str) -> ErrorRate:
    """Word error rate of `hypothesis` against `reference`."""
    return _levenshtein(tokens(reference), tokens(hypothesis))


def cer(reference: str, hypothesis: str) -> ErrorRate:
    """Character error rate. Secondary metric: degrades more gracefully than WER
    when the ASR mangles word boundaries, which time-compressed speech provokes."""
    return _levenshtein(list(normalise(reference).replace(" ", "")),
                        list(normalise(hypothesis).replace(" ", "")))
