"""One continuous voice per phrase.

Instead of separate overlapping syllable clips, a phrase is one unbroken buzz. Three tracks move
smoothly underneath it:
  - pitch: slides into each syllable's note and sags a little across it
  - mouth (formants): glides from vowel to vowel, as a real mouth can't jump (coarticulation)
  - loudness: each syllable swells; s/t/k leave short gaps; m/n/l leave short dips
This removes the vowel snaps and the brief two-pitches-at-once overlap of the clip approach.
"""

from __future__ import annotations

import zlib
from itertools import groupby

import numpy as np
from scipy.signal import lfilter

from vocoid import SR
from vocoid.synth.formants import apply_tract_tracks, formants_for
from vocoid.synth.glottal import buzz
from vocoid.synth.syllable import ONSET_DIP, SAG_SEMITONES, TARGET_RMS
from vocoid.voice import Voice

PITCH_SMOOTH = 0.035   # s: how quickly pitch slides into a new syllable
MOUTH_SMOOTH = 0.03    # s: how quickly the mouth moves to the next vowel
LOUD_SMOOTH = 0.012    # s: softness of loudness edges (scaled by the mood's attack)
CONSONANT_DIP = 0.55   # loudness drop between syllables joined by a voiced consonant (m, n, l)
DIP_WIDTH = 0.045      # s
LEVEL_EVEN = 0.8       # 0 = raw vowel levels, 1 = every vowel exactly as loud
TAIL = 0.06            # s of release after the last syllable


def _one_pole(x: np.ndarray, tau: float, sr: int, init: float | None = None) -> np.ndarray:
    """Exponential smoothing; output starts at init (default: first input value)."""
    if tau <= 0 or len(x) == 0:
        return x
    a = np.exp(-1.0 / (tau * sr))
    y0 = x[0] if init is None else init
    y, _ = lfilter([1 - a], [1, -a], x, zi=[a * y0])
    return y


def _smooth_both_ways(x: np.ndarray, tau: float, sr: int) -> np.ndarray:
    """Zero-lag smoothing (forward then backward), so edges don't arrive late."""
    return _one_pole(_one_pole(x, tau, sr)[::-1], tau, sr)[::-1]


def group_phrases(hits: list) -> list[list]:
    """Split VoiceHits into phrases (same phrase number, in time order)."""
    hits = sorted(hits, key=lambda h: h.time)
    return [list(g) for _, g in groupby(hits, key=lambda h: h.phrase)]


def synth_phrase(hits: list, voice: Voice, line: str = "", sr: int = SR) -> tuple[float, np.ndarray]:
    """Render one phrase of VoiceHits. Returns (start time in seconds, samples)."""
    hits = sorted(hits, key=lambda h: h.time)
    e = hits[0].emotion  # voice quality per phrase; pitch and loudness still vary per syllable
    rng = np.random.default_rng([zlib.crc32(line.encode()), hits[0].index, zlib.crc32(voice.name.encode())])

    t0 = hits[0].time
    end = max(h.time + h.duration for h in hits) + TAIL
    n = max(32, int(round((end - t0) * sr)))
    t = t0 + np.arange(n) / sr
    K = len(hits)
    starts = np.array([h.time for h in hits])
    seg_end = np.array([hits[i + 1].time if i + 1 < K else hits[i].time + hits[i].duration
                        for i in range(K)])
    k = np.clip(np.searchsorted(starts, t, side="right") - 1, 0, K - 1)
    u = np.clip((t - starts[k]) / np.maximum(seg_end - starts, 1e-3)[k], 0, 1)  # progress in syllable

    # pitch track (semitones vs. base pitch)
    target = 12 * np.log2([h.ratio for h in hits])
    st = target[k] - SAG_SEMITONES * e.sag * u ** 1.5
    st = _one_pole(st, PITCH_SMOOTH, sr, init=target[0] - ONSET_DIP)
    f0 = voice.base_pitch * 2 ** (st / 12)
    if voice.wobble_depth:
        f0 *= 1 + voice.wobble_depth * np.sin(2 * np.pi * voice.wobble_hz * t)
    if e.tremor_depth:
        f0 *= 1 + e.tremor_depth * np.sin(2 * np.pi * e.tremor_hz * t)

    # mouth tracks: each syllable's vowel, sliding to its second vowel for diphthongs
    def table(vowel_of):
        rows = [formants_for(vowel_of(h), voice.tube_scale, voice.mouth_closed, sr) for h in hits]
        return np.array([[f for f, _ in r] for r in rows]), np.array([[b for _, b in r] for r in rows])

    fs, bs = table(lambda h: h.vowel)
    fe, be = table(lambda h: h.vowel_end or h.vowel)
    p = np.clip((u - 0.3) / 0.5, 0, 1)
    p = (0.5 - 0.5 * np.cos(np.pi * p))[:, None]
    log_f = np.log(fs[k]) + (np.log(fe[k]) - np.log(fs[k])) * p
    bw = bs[k] + (be[k] - bs[k]) * p
    for j in range(log_f.shape[1]):
        log_f[:, j] = _one_pole(log_f[:, j], MOUTH_SMOOTH, sr)
        bw[:, j] = _one_pole(bw[:, j], MOUTH_SMOOTH, sr)

    # loudness track
    amp = np.zeros(n)
    for h in hits:
        a = int(round((h.time - t0) * sr))
        b = int(round((h.time + h.duration - t0) * sr))
        amp[a:b] = np.maximum(amp[a:b], h.gain)
    sigma = DIP_WIDTH / 2
    for i in range(1, K):
        prev = hits[i - 1]
        if prev.time + prev.duration >= hits[i].time:  # joined without a gap: voiced consonant
            c = hits[i].time - 0.01
            amp *= 1 - CONSONANT_DIP * np.exp(-0.5 * ((t - c) / sigma) ** 2)
    amp = _smooth_both_ways(amp, LOUD_SMOOTH * e.attack, sr)

    # source -> mouth
    src, flow = buzz(f0, sr, voice.jitter * e.jitter, voice.shimmer, voice.flutter, rng,
                     tension=e.tension)
    src /= np.sqrt(np.mean(src ** 2)) + 1e-12
    breath = voice.breath * e.breath
    if breath:
        noise = rng.standard_normal(n) * (0.3 + 0.7 * flow / (flow.max() + 1e-12))
        src = src + breath * noise / (np.sqrt(np.mean(noise ** 2)) + 1e-12)
    y = apply_tract_tracks(src, np.exp(log_f), bw, sr)

    # even out loudness differences between vowels (keep a little of the natural variation)
    env = np.sqrt(_smooth_both_ways(y ** 2, 0.03, sr)) + 1e-9
    y = y / env ** LEVEL_EVEN
    y *= TARGET_RMS / (np.sqrt(np.mean(y ** 2)) + 1e-12)
    return t0, (y * amp).astype(np.float32)
