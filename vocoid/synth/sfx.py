"""Small synthetic sound effects that a line can fire at one exact letter, e.g. {purr}."""

from functools import lru_cache

import numpy as np
from scipy.signal import butter, lfilter

from vocoid import SR
from vocoid.synth.envelope import swell


def purr(sr: int = SR) -> np.ndarray:
    n = int(0.55 * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(7)
    b, a = butter(2, 350 / (sr / 2))
    noise = lfilter(b, a, rng.standard_normal(n))
    flutter = (0.5 + 0.5 * np.sin(2 * np.pi * 24 * t)) ** 2
    y = noise * flutter * swell(n, sr, 0.06, 0.15)
    return (0.5 * y / (np.max(np.abs(y)) + 1e-12)).astype(np.float32)


def chime(sr: int = SR) -> np.ndarray:
    n = int(0.4 * sr)
    t = np.arange(n) / sr
    y = (np.sin(2 * np.pi * 1320 * t) + 0.5 * np.sin(2 * np.pi * 1980 * t)) * np.exp(-t * 9)
    return (0.35 * y / np.max(np.abs(y))).astype(np.float32)


SFX = {"purr": purr, "chime": chime}


@lru_cache(maxsize=32)
def get_sfx(name: str, sr: int = SR) -> np.ndarray:
    return SFX[name](sr)
