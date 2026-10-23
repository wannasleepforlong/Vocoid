"""Speech rhythm: when each syllable sounds and when each letter appears.

Timed like speech, not like spelling: a syllable's length comes from its consonants and stress,
words in a phrase run together, and pauses come only from punctuation. Letters are then spread
over that timeline, so each syllable's letter still appears exactly as its sound starts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from typing import Callable

from vocoid.emotion import NEUTRAL, Emotion
from vocoid.text.prosody import Note, Phrase
from vocoid.text.syllables import Syllable

SYL_BASE = 0.13          # seconds, a bare vowel syllable
PER_CONSONANT = 0.035    # each consonant around it adds a little
STRESSED = 1.25
UNSTRESSED = 0.85
FUNCTION = 0.75          # "the", "to": quick
LONG_VOWEL = 0.03        # ee, ooh, ah, aw, er and slides are held a bit longer
LONG_VOWELS = {"ee", "ooh", "ah", "aw", "er"}
PHRASE_FINAL = 1.4       # last syllable before a pause is drawn out
PAUSE = {",": 0.22, ";": 0.30, ":": 0.30, ".": 0.45, "!": 0.45, "?": 0.45, "...": 0.65, "": 0.0}
VOICELESS_GAP = 0.025    # s, t, k... are silent in a vowel-only voice: leave a short gap
MAX_GAP = 0.06
OVERLAP = 0.03           # sound runs into the next one a little, so it stays one stream
MIN_SOUND = 0.07


@dataclass(frozen=True)
class Sound:
    index: int
    time: float
    duration: float


def syllable_length(s: Syllable, final: bool) -> float:
    d = SYL_BASE + PER_CONSONANT * (len(s.onset) + len(s.coda))
    if s.function_word:
        d *= FUNCTION
    elif s.stress:
        d *= STRESSED
    else:
        d *= UNSTRESSED
    if s.vowel_end or s.vowel in LONG_VOWELS:
        d += LONG_VOWEL
    if final:
        d *= PHRASE_FINAL
    return d


def timeline(text: str, syllables: list[Syllable], notes: dict[int, Note],
             phrases: list[Phrase], speed: float = 1.0,
             mood_at: Callable[[int], Emotion] | None = None) -> tuple[dict[int, Sound], list[float]]:
    """Returns (sound per syllable char index, reveal time per char)."""
    mood_at = mood_at or (lambda i: NEUTRAL)
    by_index = {s.index: s for s in syllables}
    sounds: dict[int, Sound] = {}
    anchors: list[tuple[int, float]] = []
    t = 0.0
    for ph in phrases:
        idx = sorted(i for i in notes if ph.start <= i < ph.end)
        for i in idx:
            s = by_index[i]
            sp = speed * mood_at(i).rate
            d = syllable_length(s, notes[i].final) / sp
            start = t + min(MAX_GAP, VOICELESS_GAP * s.voiceless_onset) / sp
            end = t + d - min(MAX_GAP, VOICELESS_GAP * s.voiceless_coda) / sp + OVERLAP
            sounds[i] = Sound(i, start, max(MIN_SOUND, end - start))
            anchors.append((i, start))
            t += d
        if ph.end < len(text):
            anchors.append((ph.end, t))  # punctuation shows as the phrase finishes, then the pause
        mood = mood_at(max(ph.start, ph.end - 1))
        t += PAUSE.get(ph.kind, 0.0) * mood.pause / (speed * mood.rate)

    n = len(text)
    if not n:
        return sounds, []
    if not anchors or anchors[0][0] > 0:
        anchors.insert(0, (0, 0.0))
    if anchors[-1][0] < n - 1:
        anchors.append((n - 1, max(anchors[-1][1], t)))
    xs, ts = zip(*sorted(dict(anchors).items()))
    times = np.interp(np.arange(n), xs, np.maximum.accumulate(ts))
    return sounds, [float(x) for x in times]
