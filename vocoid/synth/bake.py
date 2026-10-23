"""Vowel bank: each voice's six vowels at its base pitch. Used for export and inspection.

Playback no longer uses the bank: syllables are synthesized live (see syllable.py) so pitch
can move naturally. One syllable takes about a millisecond to render.
"""

from __future__ import annotations

import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from vocoid import SR
from vocoid.synth.formants import VOWELS
from vocoid.synth.syllable import synth_syllable
from vocoid.voice import Voice

BANK_SECONDS = 0.24


def synth_vowel(vowel: str, voice: Voice, sr: int = SR, duration: float = BANK_SECONDS,
                rng: np.random.Generator | None = None, f0: float | None = None,
                envelope: bool = True) -> np.ndarray:
    """Steady-pitch vowel (no glide/sag/flutter). f0 overrides the voice's base pitch."""
    ratio = (f0 / voice.base_pitch) if f0 else 1.0
    return synth_syllable(vowel, voice, ratio, duration, sr, rng, steady=True, envelope=envelope)


@lru_cache(maxsize=64)
def get_bank(voice: Voice, sr: int = SR) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(zlib.crc32(voice.name.encode()))
    return {v: synth_syllable(v, voice, 1.0, BANK_SECONDS, sr, rng) for v in VOWELS}


def to_int16(samples: np.ndarray) -> np.ndarray:
    return (np.clip(samples, -1, 1) * 32767).astype(np.int16)


def save_bank(voice: Voice, folder: str | Path, sr: int = SR) -> int:
    """Write one WAV per vowel. Returns total bytes of audio data."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    total = 0
    for vowel, y in get_bank(voice, sr).items():
        data = to_int16(y)
        wavfile.write(folder / f"{voice.name}_{vowel}.wav", sr, data)
        total += data.nbytes
    return total
