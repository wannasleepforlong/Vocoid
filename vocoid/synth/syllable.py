"""One syllable, synthesized live with a moving pitch.

Why live instead of pitch-shifting a baked sample:
  - pitch can glide in from the last syllable and sag a little, like real speech
  - mouth shape (formants) stays put when pitch changes, so no chipmunk effect
  - every syllable gets fresh jitter, never the same recording twice
  - mood can change voice quality (breath, tension, tremor) per syllable
"""

from __future__ import annotations

import zlib

import numpy as np

from vocoid import SR
from vocoid.emotion import NEUTRAL, Emotion
from vocoid.synth.envelope import swell
from vocoid.synth.formants import apply_tract
from vocoid.synth.glottal import buzz
from vocoid.voice import Voice

TARGET_RMS = 0.12
GLIDE_SECONDS = 0.06    # time to slide from the previous syllable's pitch
SAG_SEMITONES = 1.0     # natural pitch sag across one syllable
ONSET_DIP = 0.8         # first syllable of a phrase starts this many semitones low, then rises in
ATTACK = 0.03
RELEASE = 0.05


def pitch_curve(voice: Voice, ratio: float, n: int, sr: int, t0: float = 0.0,
                glide_from: float | None = None, steady: bool = False,
                emotion: Emotion = NEUTRAL) -> np.ndarray:
    """Hz per sample. ratio / glide_from are relative to voice.base_pitch."""
    t = np.arange(n) / sr
    target = 12 * np.log2(ratio)
    sag = SAG_SEMITONES * emotion.sag
    if steady:
        st = np.full(n, target)
    else:
        start = (12 * np.log2(glide_from) - sag) if glide_from else target - ONSET_DIP
        g = np.clip(t / GLIDE_SECONDS, 0, 1)
        g = 0.5 - 0.5 * np.cos(np.pi * g)
        dur = max(t[-1], 1e-6) if n else 1.0
        st = start + (target - start) * g - sag * (t / dur) ** 1.5
    f0 = voice.base_pitch * 2 ** (st / 12)
    if voice.wobble_depth:
        f0 *= 1 + voice.wobble_depth * np.sin(2 * np.pi * voice.wobble_hz * (t0 + t))
    if emotion.tremor_depth:
        f0 *= 1 + emotion.tremor_depth * np.sin(2 * np.pi * emotion.tremor_hz * (t0 + t))
    return f0


def synth_syllable(vowel: str, voice: Voice, ratio: float = 1.0, duration: float = 0.24,
                   sr: int = SR, rng: np.random.Generator | None = None, t0: float = 0.0,
                   glide_from: float | None = None, steady: bool = False,
                   envelope: bool = True, vowel_end: str | None = None,
                   emotion: Emotion = NEUTRAL) -> np.ndarray:
    """Buzz (with breath pulsed by the airflow) -> mouth tube -> swell envelope.
    vowel_end makes a diphthong: the mouth slides from vowel to vowel_end ("right" = ah -> ee).
    emotion shapes voice quality; its pitch level and loudness are applied by the engine."""
    rng = rng or np.random.default_rng(0)
    e = emotion
    n = max(16, int(duration * sr))
    f0 = pitch_curve(voice, ratio, n, sr, t0, glide_from, steady, e)
    src, flow = buzz(f0, sr, voice.jitter * e.jitter, voice.shimmer,
                     0.0 if steady else voice.flutter, rng, tension=e.tension)
    src /= np.sqrt(np.mean(src ** 2)) + 1e-12
    breath = voice.breath * e.breath
    if breath:
        # breath leaks mostly while the folds are open, so it pulses with the voice
        noise = rng.standard_normal(n) * (0.3 + 0.7 * flow / (flow.max() + 1e-12))
        src = src + breath * noise / (np.sqrt(np.mean(noise ** 2)) + 1e-12)
    y = apply_tract(src, vowel, sr, voice.tube_scale, voice.mouth_closed, vowel_end)
    y *= TARGET_RMS / (np.sqrt(np.mean(y ** 2)) + 1e-12)
    if envelope:
        y *= swell(n, sr, min(ATTACK * e.attack, duration * 0.4), min(RELEASE, duration * 0.4))
    return y.astype(np.float32)


def hit_audio(hit, voice: Voice, line: str, sr: int = SR) -> np.ndarray:
    """Audio for one engine VoiceHit. Seeded by line + letter, so a line always sounds the same."""
    seed = [zlib.crc32(line.encode()), hit.index, zlib.crc32(voice.name.encode())]
    y = synth_syllable(hit.vowel, voice, hit.ratio, hit.duration, sr,
                       np.random.default_rng(seed), hit.time, hit.glide_from,
                       vowel_end=hit.vowel_end, emotion=hit.emotion)
    return y * hit.gain
