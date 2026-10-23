"""The tube: throat + mouth boost three pitch bands (formants). Which bands = which vowel."""

import numpy as np
from scipy.signal import lfilter

# F1, F2, F3 in Hz. Adult male averages from classic speech research (Peterson & Barney 1952 style).
VOWELS: dict[str, tuple[float, float, float]] = {
    "ah": (730, 1090, 2440),   # father
    "ae": (660, 1720, 2410),   # cat
    "eh": (530, 1840, 2480),   # bed
    "ih": (390, 1990, 2550),   # sit
    "ee": (270, 2290, 3010),   # see
    "aw": (570, 840, 2410),    # thought
    "oh": (500, 900, 2400),    # go
    "uu": (440, 1020, 2240),   # book
    "ooh": (300, 870, 2240),   # boot
    "uh": (640, 1190, 2390),   # but
    "er": (490, 1350, 1690),   # her
}
BANDWIDTHS = (80.0, 100.0, 140.0)
# F4, F5: roughly the same for every vowel. Without them everything above ~3 kHz collapses
# (three cascaded resonators roll off steeply) and the voice sounds muffled.
HIGHER_FORMANTS = ((3500.0, 200.0), (4500.0, 250.0))

NEUTRAL_F2 = 1150.0  # where F2 drifts when the mouth barely opens
BLOCK = 32           # samples per filter update when formants move


def resonator_coefs(freq: float, bw: float, sr: int) -> tuple[float, float, float]:
    """Klatt two-pole resonator: boosts a band around freq."""
    T = 1.0 / sr
    c = -np.exp(-2 * np.pi * bw * T)
    b = 2 * np.exp(-np.pi * bw * T) * np.cos(2 * np.pi * freq * T)
    return 1 - b - c, b, c


def resonator(x: np.ndarray, freq: float, bw: float, sr: int) -> np.ndarray:
    a, b, c = resonator_coefs(freq, bw, sr)
    return lfilter([a], [1, -b, -c], x)


def formants_for(vowel: str, tube_scale: float = 1.0, mouth_closed: float = 0.0,
                 sr: int | None = None) -> list[tuple[float, float]]:
    """Formant (freq, bandwidth) list for a vowel and tract shape.

    tube_scale > 1 = shorter tube (woman, child): all bands move up together.
    mouth_closed 0..1 = mutter: F1 drops, F2 pulls to neutral, bands get wider (muffled).
    """
    f1, f2, f3 = VOWELS[vowel]
    m = mouth_closed
    f1 *= 1 - 0.35 * m
    f2 += (NEUTRAL_F2 - f2) * 0.45 * m
    f3 *= 1 - 0.05 * m
    out = []
    bands = list(zip((f1, f2, f3), BANDWIDTHS)) + list(HIGHER_FORMANTS)
    for f, bw in bands:
        f *= tube_scale
        bw *= (1 + 0.8 * m) * (0.5 + 0.5 * tube_scale)
        if sr is not None:
            f = min(f, 0.45 * sr)
        out.append((f, bw))
    return out


def apply_tract(x: np.ndarray, vowel: str, sr: int, tube_scale: float = 1.0,
                mouth_closed: float = 0.0, vowel_end: str | None = None,
                glide: tuple[float, float] = (0.3, 0.8)) -> np.ndarray:
    """Pass the buzz through five cascaded resonators (F1-F3 = vowel, F4-F5 = presence).

    With vowel_end, the mouth slides from vowel to vowel_end between glide[0] and glide[1]
    of the sound (e.g. "right" = ah -> ee). Filters update every BLOCK samples, keeping state.
    """
    start = formants_for(vowel, tube_scale, mouth_closed, sr)
    if vowel_end is None or vowel_end == vowel:
        y = x
        for f, bw in start:
            y = resonator(y, f, bw, sr)
        return y

    end = formants_for(vowel_end, tube_scale, mouth_closed, sr)
    n = len(x)
    y = np.empty(n)
    zi = [np.zeros(2) for _ in start]
    g0, g1 = glide
    for s in range(0, n, BLOCK):
        p = np.clip(((s + BLOCK / 2) / n - g0) / (g1 - g0), 0, 1)
        p = 0.5 - 0.5 * np.cos(np.pi * p)
        blk = x[s:s + BLOCK]
        for k, ((fa, ba), (fb, bb)) in enumerate(zip(start, end)):
            # move in log-frequency so the slide sounds even
            f = fa * (fb / fa) ** p
            bw = ba + (bb - ba) * p
            a, b, c = resonator_coefs(f, bw, sr)
            blk, zi[k] = lfilter([a], [1, -b, -c], blk, zi=zi[k])
        y[s:s + BLOCK] = blk
    return y
