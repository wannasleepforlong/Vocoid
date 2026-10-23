"""The buzz: what the vocal folds make before the mouth shapes it.

Pitch is a smooth curve, not fixed whole-sample periods. A continuous phase runs through the
pulse shape, so pitch can glide and never steps. Rendered 4x oversampled so high voices
don't alias into metallic tones.
"""

import numpy as np
from scipy.signal import lfilter, resample_poly

OVERSAMPLE = 4
OPEN_Q = 0.5    # fraction of a cycle the folds spend opening
CLOSE_Q = 0.12  # fraction spent snapping shut (fast close = the bright edge of the buzz)


def glottal_flow(phase: np.ndarray, open_q: float = OPEN_Q, close_q: float = CLOSE_Q) -> np.ndarray:
    """Rosenberg airflow pulse for cycle phase in [0, 1): slow open, fast close, then shut."""
    g = np.zeros_like(phase)
    rise = phase < open_q
    g[rise] = 0.5 * (1 - np.cos(np.pi * phase[rise] / open_q))
    fall = (phase >= open_q) & (phase < open_q + close_q)
    g[fall] = np.cos(0.5 * np.pi * (phase[fall] - open_q) / close_q)
    return g


def rosenberg_pulse(n: int, open_q: float = OPEN_Q, close_q: float = CLOSE_Q) -> np.ndarray:
    """One cycle of n samples (kept for tools and tests)."""
    return glottal_flow(np.arange(n) / n, open_q, close_q)


def smooth_noise(n: int, sr: float, rate_hz: float, rng: np.random.Generator) -> np.ndarray:
    """Slow random drift: random points rate_hz apart, cosine-blended. Roughly unit std."""
    x = np.arange(n) / sr * rate_hz
    pts = rng.standard_normal(int(x[-1]) + 3 if n else 3)
    i = np.floor(x).astype(int)
    w = 0.5 - 0.5 * np.cos(np.pi * (x - i))
    return pts[i] * (1 - w) + pts[i + 1] * w


TILT_HZ = 3000.0
LAX_TILT_DB = 15.0    # fully lax voice: this much softer at 3 kHz (folds close gently)
PRESSED_TILT_DB = 6.0  # fully pressed voice: this much brighter at 3 kHz


def _solve(fn, lo: float, hi: float, target: float) -> float:
    """Bisection for an increasing fn."""
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def spectral_tilt(x: np.ndarray, sr: int, tension: float) -> np.ndarray:
    """Lax voices lose high frequencies, pressed voices gain them (like Klatt's tilt parameter).
    DC gain stays 1, so only the balance of lows vs. highs changes."""
    if abs(tension) < 1e-3:
        return x
    w = 2 * np.pi * TILT_HZ / sr
    if tension < 0:
        want = 10 ** (-LAX_TILT_DB * min(1.0, -tension) / 20)
        # one-pole lowpass y = (1-b) x + b y[-1]; attenuation at w grows with b
        att = lambda b: -abs((1 - b) / (1 - b * np.exp(-1j * w)))
        b = _solve(att, 0.0, 0.999, -want)
        return lfilter([1 - b], [1, -b], x)
    want = 10 ** (PRESSED_TILT_DB * min(1.0, tension) / 20)
    # y = x + k (x - x[-1]); boost at w grows with k
    gain = lambda k: abs(1 + k * (1 - np.exp(-1j * w)))
    k = _solve(gain, 0.0, 10.0, want)
    return lfilter([1 + k, -k], [1], x)


def tension_shape(tension: float) -> tuple[float, float]:
    """-1 lax (soft, dark) .. +1 pressed (harsh, bright) -> (open_q, close_q)."""
    t = max(-1.0, min(1.0, tension))
    return OPEN_Q - 0.1 * t, max(0.04, CLOSE_Q - 0.06 * t)


def buzz(f0, sr: int, jitter: float = 0.006, shimmer: float = 0.03, flutter: float = 0.008,
         rng: np.random.Generator | None = None, duration: float | None = None,
         tension: float = 0.0):
    """Glottal source following a pitch curve.

    f0: Hz per output sample (array), or a constant with `duration`.
    jitter: per-cycle pitch randomness. shimmer: per-cycle loudness randomness.
    flutter: slow pitch drift (~5 Hz), the natural unsteadiness of a held note.
    Returns (source, flow): source = the sharp buzz we hear, flow = airflow (0..1, used to pulse breath).
    """
    rng = rng or np.random.default_rng(0)
    if np.isscalar(f0):
        f0 = np.full(int(duration * sr), float(f0))
    f0 = np.asarray(f0, dtype=float)
    n = len(f0)
    hr = sr * OVERSAMPLE
    nh = n * OVERSAMPLE
    f = np.interp(np.arange(nh) / OVERSAMPLE, np.arange(n), f0)
    if flutter:
        f = f * (1 + flutter * smooth_noise(nh, hr, 5.0, rng))

    # per-cycle jitter: find cycle numbers, then give each cycle its own small pitch error
    cyc = np.floor(np.cumsum(f / hr)).astype(int)
    f = f * (1 + jitter * rng.standard_normal(cyc[-1] + 2))[cyc]
    phase = np.cumsum(f / hr)
    cyc = np.floor(phase).astype(int)
    flow = glottal_flow(phase - cyc, *tension_shape(tension))
    flow *= (1 + shimmer * rng.standard_normal(cyc[-1] + 2))[cyc]

    src = np.diff(flow, prepend=0.0)  # flow derivative = the buzz
    src = spectral_tilt(resample_poly(src, 1, OVERSAMPLE)[:n], sr, tension)
    flow = resample_poly(flow, 1, OVERSAMPLE)[:n]
    src /= np.max(np.abs(src)) + 1e-12
    return src, np.clip(flow, 0, None)
