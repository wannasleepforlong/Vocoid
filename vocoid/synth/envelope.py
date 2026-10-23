"""Volume shape of one syllable: swell in, hold, let go (a voice, not a drum hit)."""

import numpy as np


def swell(n: int, sr: int, attack: float = 0.035, release: float = 0.08) -> np.ndarray:
    env = np.ones(n)
    a = min(n // 2, int(attack * sr))
    r = min(n - a, int(release * sr))
    if a > 0:
        env[:a] = 0.5 * (1 - np.cos(np.pi * np.arange(a) / a))
    if r > 0:
        env[n - r:] = 0.5 * (1 + np.cos(np.pi * np.arange(r) / r))
    return env


def drum_hit(n: int, sr: int, decay: float = 0.04) -> np.ndarray:
    """The wrong shape (loud then fade). Kept for comparison."""
    return np.exp(-np.arange(n) / (decay * sr))
