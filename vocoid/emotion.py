"""Emotions: how a mood bends a voice. Values follow what speech research reports for acted emotion
(e.g. Murray & Arnott 1993): pitch level + range, tempo, loudness, voice quality.

Tag a line with {mood:sad} or {mood:happy:0.5} (0..1 strength). Tags can switch mood mid-line.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace


@dataclass(frozen=True)
class Emotion:
    name: str = "neutral"
    pitch: float = 1.0        # overall pitch level (multiplier)
    range: float = 1.0        # how far the tune moves (multiplier)
    rate: float = 1.0         # talking speed (multiplier)
    gain: float = 1.0         # loudness (multiplier)
    breath: float = 1.0       # breathiness (multiplier)
    jitter: float = 1.0       # pitch roughness (multiplier)
    tremor_hz: float = 0.0    # shaky voice (scared)
    tremor_depth: float = 0.0
    tension: float = 0.0      # -1 lax/soft .. +1 pressed/harsh (vocal fold closing speed)
    sag: float = 1.0          # pitch sag inside each syllable (multiplier; sad sags more)
    stress: float = 1.0       # strength of word accents (multiplier; angry hits harder)
    pause: float = 1.0        # length of punctuation pauses (multiplier)
    onset_lift: float = 0.0   # extra pitch at the start of each phrase
    final_rise: float = 0.0   # pitch offset on the last syllable of a phrase (+ lilt, - drop)
    attack: float = 1.0       # syllable fade-in time (multiplier; angry = abrupt)

    def blend(self, amount: float) -> "Emotion":
        """amount 0 = neutral, 1 = full emotion."""
        a = max(0.0, min(1.0, amount))
        base = NEUTRAL
        vals = {}
        for f in fields(self):
            if f.name == "name":
                continue
            v0, v1 = getattr(base, f.name), getattr(self, f.name)
            vals[f.name] = v0 + (v1 - v0) * a
        return replace(self, **vals)


NEUTRAL = Emotion()

EMOTIONS: dict[str, Emotion] = {e.name: e for e in [
    NEUTRAL,
    Emotion("happy", pitch=1.12, range=1.5, rate=1.08, gain=1.1, breath=0.9, tension=0.2,
            sag=0.8, stress=1.3, onset_lift=0.05, final_rise=0.06),
    Emotion("sad", pitch=0.9, range=0.45, rate=0.78, gain=0.75, breath=1.8, jitter=1.3,
            tension=-0.6, sag=1.8, stress=0.6, pause=1.5, final_rise=-0.08, attack=1.8),
    Emotion("angry", pitch=1.04, range=1.3, rate=1.12, gain=1.4, breath=0.6, tension=0.9,
            sag=1.2, stress=1.7, pause=0.8, final_rise=-0.06, attack=0.5),
    Emotion("scared", pitch=1.2, range=1.2, rate=1.25, gain=0.9, breath=1.5, jitter=2.5,
            tremor_hz=6.0, tremor_depth=0.025, tension=0.3, pause=0.7),
    Emotion("surprised", pitch=1.15, range=1.8, rate=0.95, gain=1.15, stress=1.4,
            onset_lift=0.2, final_rise=0.1),
    Emotion("tired", pitch=0.94, range=0.35, rate=0.75, gain=0.8, breath=1.5, tension=-0.4,
            sag=1.5, stress=0.5, pause=1.4, attack=1.5),
]}


def get_emotion(name: str | Emotion | None, amount: float = 1.0) -> Emotion:
    if name is None:
        return NEUTRAL
    if isinstance(name, Emotion):
        return name.blend(amount) if amount != 1.0 else name
    try:
        e = EMOTIONS[name.lower()]
    except KeyError:
        raise KeyError(f"unknown mood {name!r}; choose from {', '.join(EMOTIONS)}") from None
    return e.blend(amount) if amount != 1.0 else e
