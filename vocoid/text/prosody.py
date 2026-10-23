"""The tune, all from punctuation already in the line.

Pitch offset per syllable = phrase drift + tiny random nudge (+ caps shout).
The engine adds the voice's base pitch and the old-age wobble on top.
"""

import zlib
from dataclasses import dataclass

import numpy as np

from vocoid.text.syllables import Syllable, find_syllables, words

BREAKS = ",;:.?!…"

DRIFT_START = 0.12    # phrase starts a bit high...
DRIFT_SPAN = 0.26     # ...and drifts down across it (~4 semitones, like real speech)
NUDGE = 0.04
QUESTION_RISE = 0.30  # last word of a question climbs
EXCLAIM_LIFT = 0.18   # exclamation starts high
EXCLAIM_GAIN = 1.15
TRAIL_FADE = 0.7      # "..." fades out
SHOUT_PITCH = 0.20
SHOUT_GAIN = 1.45
STRESS_PITCH = 0.06   # stressed syllable gets a small pitch accent...
STRESS_GAIN = 1.1     # ...and is a bit louder
UNSTRESS_GAIN = 0.85
FUNCTION_GAIN = 0.8   # "the", "to", "of" are tucked in quietly


@dataclass(frozen=True)
class Note:
    index: int
    vowel: str
    pitch: float  # offset added to 1.0 (scaled by voice.pitch_range)
    gain: float
    phrase: int = 0      # phrase number in the line
    final: bool = False  # last syllable before a pause


@dataclass(frozen=True)
class Phrase:
    start: int
    end: int   # exclusive, punctuation not included
    kind: str  # "?", "!", "...", ".", ",", or "" (line ends without punctuation)


def split_phrases(text: str) -> list[Phrase]:
    phrases = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        if text[i] in BREAKS:
            j = i
            while j < n and text[j] in BREAKS:
                j += 1
            run = text[i:j]
            if "?" in run:
                kind = "?"
            elif "..." in run or "…" in run:
                kind = "..."
            elif "!" in run:
                kind = "!"
            else:
                kind = run[0] if run[0] in ",;:" else "."
            phrases.append(Phrase(start, i, kind))
            start = i = j
        else:
            i += 1
    if start < n and text[start:].strip():
        phrases.append(Phrase(start, n, ""))
    return phrases


def plan_line(text: str, syllables: list[Syllable] | None = None) -> dict[int, Note]:
    """Map char index -> Note for every char that should make a sound."""
    if syllables is None:
        syllables = find_syllables(text)
    rng = np.random.default_rng(zlib.crc32(text.encode()))  # same line, same tune
    shout_words = {i for i, (_, _, w) in enumerate(words(text)) if len(w) >= 2 and w.isupper()}

    notes: dict[int, Note] = {}
    for ph_i, ph in enumerate(split_phrases(text)):
        sy = [s for s in syllables if ph.start <= s.index < ph.end]
        if not sy:
            continue
        last_word = sy[-1].word
        last_word_sy = [s for s in sy if s.word == last_word]
        for k, s in enumerate(sy):
            p = k / (len(sy) - 1) if len(sy) > 1 else 0.0
            pitch = DRIFT_START - DRIFT_SPAN * p
            gain = 1.0
            if ph.kind == "!":
                pitch += EXCLAIM_LIFT * (1 - p)
                gain *= EXCLAIM_GAIN
            elif ph.kind == "?" and s.word == last_word:
                j = last_word_sy.index(s)
                pitch += QUESTION_RISE * (j + 1) / len(last_word_sy)
            elif ph.kind == "...":
                gain *= 1 - TRAIL_FADE * p
                pitch -= 0.08 * p
            if s.function_word:
                gain *= FUNCTION_GAIN
            elif s.stress:
                pitch += STRESS_PITCH
                gain *= STRESS_GAIN
            else:
                gain *= UNSTRESS_GAIN
            if s.word in shout_words:
                pitch += SHOUT_PITCH
                gain *= SHOUT_GAIN
            pitch += rng.uniform(-NUDGE, NUDGE)
            notes[s.index] = Note(s.index, s.vowel, pitch, gain, ph_i, k == len(sy) - 1)
    return notes
