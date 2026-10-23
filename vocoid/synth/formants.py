"""The tube: throat + mouth boost pitch bands (formants). Which bands = which vowel.

Men, women and children get their own measured vowel tables. A shorter tube is not just a uniform
shift: e.g. a woman's "cat" F1 is ~30% above a man's, her "father" F1 only ~16%. Getting this
right is a big part of why a voice reads as a woman or a child rather than "a man, pitched up".
"""

import numpy as np
from scipy.signal import lfilter

# F1, F2, F3 in Hz: Peterson & Barney (1952) averages per speaker group.
# "oh" (go) is not in that study: scaled from "aw" with the same group ratios.
VOWELS_BY_KIND: dict[str, dict[str, tuple[float, float, float]]] = {
    "man": {
        "ee": (270, 2290, 3010), "ih": (390, 1990, 2550), "eh": (530, 1840, 2480),
        "ae": (660, 1720, 2410), "ah": (730, 1090, 2440), "aw": (570, 840, 2410),
        "oh": (500, 900, 2400), "uu": (440, 1020, 2240), "ooh": (300, 870, 2240),
        "uh": (640, 1190, 2390), "er": (490, 1350, 1690),
    },
    "woman": {
        "ee": (310, 2790, 3310), "ih": (430, 2480, 3070), "eh": (610, 2330, 2990),
        "ae": (860, 2050, 2850), "ah": (850, 1220, 2810), "aw": (590, 920, 2710),
        "oh": (520, 990, 2700), "uu": (470, 1160, 2680), "ooh": (370, 950, 2670),
        "uh": (760, 1400, 2780), "er": (500, 1640, 1960),
    },
    "child": {
        "ee": (370, 3200, 3730), "ih": (530, 2730, 3600), "eh": (690, 2610, 3570),
        "ae": (1010, 2320, 3320), "ah": (1030, 1370, 3170), "aw": (680, 1060, 3180),
        "oh": (600, 1135, 3165), "uu": (560, 1410, 3310), "ooh": (430, 1170, 3260),
        "uh": (850, 1590, 3360), "er": (560, 1820, 2160),
    },
}
VOWELS = VOWELS_BY_KIND["man"]  # vowel names; kept for callers that only need the list
KINDS = tuple(VOWELS_BY_KIND)

BANDWIDTHS = (80.0, 100.0, 140.0)
# Smaller, softer tracts damp resonances more: wider bands (less "ringing", more breathy).
BANDWIDTH_SCALE = {"man": 1.0, "woman": 1.15, "child": 1.3}
# F4, F5: roughly the same for every vowel, scaled with tract size. Without them everything
# above ~3 kHz collapses (cascaded resonators roll off steeply) and the voice sounds muffled.
HIGHER_FORMANTS = {
    "man": ((3500.0, 200.0), (4500.0, 250.0)),
    "woman": ((4000.0, 230.0), (5000.0, 290.0)),
    "child": ((4500.0, 260.0), (5600.0, 320.0)),
}

NEUTRAL_F2 = 1150.0  # where F2 drifts when the mouth barely opens (man; scaled for others)
BLOCK = 32           # samples per filter update for a single-syllable diphthong
TRACK_FRAME = 0.005  # seconds per filter update for phrase-long formant tracks

# Presence: a five-pole cascade loses far too much above ~2 kHz compared with a real voice
# (Klatt's "higher-pole correction" problem). Without this, upper formants, which carry who is
# speaking, are 15-40 dB too quiet and every voice sounds like the same muffled hum.
PRE_EMPHASIS = 0.8
SHELF_HZ = 3000.0
SHELF_DB = 15.0


def resonator_coefs(freq: float, bw: float, sr: int) -> tuple[float, float, float]:
    """Klatt two-pole resonator: boosts a band around freq."""
    T = 1.0 / sr
    c = -np.exp(-2 * np.pi * bw * T)
    b = 2 * np.exp(-np.pi * bw * T) * np.cos(2 * np.pi * freq * T)
    return 1 - b - c, b, c


def resonator(x: np.ndarray, freq: float, bw: float, sr: int) -> np.ndarray:
    a, b, c = resonator_coefs(freq, bw, sr)
    return lfilter([a], [1, -b, -c], x)


def presence(y: np.ndarray, sr: int) -> np.ndarray:
    """Pre-emphasis + high shelf (RBJ, slope 1): brings upper formants up to speech-like levels."""
    y = lfilter([1, -PRE_EMPHASIS], [1], y)
    A = 10 ** (SHELF_DB / 40)
    w = 2 * np.pi * min(SHELF_HZ, 0.4 * sr) / sr
    alpha = np.sin(w) / 2 * np.sqrt(2)
    c, sq = np.cos(w), np.sqrt(A)
    b = [A * ((A + 1) + (A - 1) * c + 2 * sq * alpha), -2 * A * ((A - 1) + (A + 1) * c),
         A * ((A + 1) + (A - 1) * c - 2 * sq * alpha)]
    a = [(A + 1) - (A - 1) * c + 2 * sq * alpha, 2 * ((A - 1) - (A + 1) * c),
         (A + 1) - (A - 1) * c - 2 * sq * alpha]
    return lfilter(b, a, y)


def formants_for(vowel: str, tube_scale: float = 1.0, mouth_closed: float = 0.0,
                 sr: int | None = None, kind: str = "man") -> list[tuple[float, float]]:
    """Formant (freq, bandwidth) list for a vowel, speaker kind and tract shape.

    kind: "man", "woman" or "child" (measured tables). tube_scale fine-tunes size within a kind
    (>1 = smaller). mouth_closed 0..1 = mutter: F1 drops, F2 pulls to neutral, bands widen.
    """
    table = VOWELS_BY_KIND[kind]
    f1, f2, f3 = table[vowel]
    m = mouth_closed
    neutral = NEUTRAL_F2 * table["uh"][1] / VOWELS_BY_KIND["man"]["uh"][1]
    f1 *= 1 - 0.35 * m
    f2 += (neutral - f2) * 0.45 * m
    f3 *= 1 - 0.05 * m
    bw_scale = BANDWIDTH_SCALE[kind]
    out = []
    bands = list(zip((f1, f2, f3), BANDWIDTHS)) + list(HIGHER_FORMANTS[kind])
    for f, bw in bands:
        f *= tube_scale
        bw *= bw_scale * (1 + 0.8 * m) * (0.5 + 0.5 * tube_scale)
        if sr is not None:
            f = min(f, 0.45 * sr)
        out.append((f, bw))
    return out


def apply_tract_tracks(x: np.ndarray, freqs: np.ndarray, bws: np.ndarray, sr: int) -> np.ndarray:
    """Cascade resonators whose formants follow per-sample tracks (n x 5), so the mouth can move
    continuously. Coefficients update every 5 ms (like classic Klatt synthesizers); tracks are
    already smoothed far slower than that. Filter state carries across blocks."""
    n, nf = freqs.shape
    y = np.empty(n)
    zi = [np.zeros(2) for _ in range(nf)]
    step = max(16, int(TRACK_FRAME * sr))
    for s in range(0, n, step):
        blk = x[s:s + step]
        mid = min(n - 1, s + step // 2)
        for k in range(nf):
            a, b, c = resonator_coefs(freqs[mid, k], bws[mid, k], sr)
            blk, zi[k] = lfilter([a], [1, -b, -c], blk, zi=zi[k])
        y[s:s + step] = blk
    return presence(y, sr)


def apply_tract(x: np.ndarray, vowel: str, sr: int, tube_scale: float = 1.0,
                mouth_closed: float = 0.0, vowel_end: str | None = None,
                glide: tuple[float, float] = (0.3, 0.8), kind: str = "man") -> np.ndarray:
    """Pass the buzz through five cascaded resonators (F1-F3 = vowel, F4-F5 = presence).

    With vowel_end, the mouth slides from vowel to vowel_end between glide[0] and glide[1]
    of the sound (e.g. "right" = ah -> ee). Filters update every BLOCK samples, keeping state.
    """
    start = formants_for(vowel, tube_scale, mouth_closed, sr, kind)
    if vowel_end is None or vowel_end == vowel:
        y = x
        for f, bw in start:
            y = resonator(y, f, bw, sr)
        return presence(y, sr)

    end = formants_for(vowel_end, tube_scale, mouth_closed, sr, kind)
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
    return presence(y, sr)
