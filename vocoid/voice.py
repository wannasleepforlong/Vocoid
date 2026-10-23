"""A voice = speaker kind + buzz pitch + a few quirks. New character = one line.

What makes a voice read as a man, woman, child or old person (beyond pitch):
  - kind: measured vowel tables per group (a woman is not "a man, pitched up")
  - tension: women's and children's voices are softer and breathier (lax), men's firmer
  - old age: rough, uneven buzz (roughness, jitter, shimmer), more breath, slow wobble,
    narrower and slower speech
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Voice:
    name: str
    base_pitch: float          # Hz of the buzz
    kind: str = "man"          # "man", "woman" or "child": which vowel table to use
    tube_scale: float = 1.0    # fine-tunes tract size within the kind (>1 = smaller)
    mouth_closed: float = 0.0  # 0..1, mutter with mouth almost shut
    jitter: float = 0.006      # per-cycle pitch randomness (healthy adults ~0.5-1%)
    shimmer: float = 0.03      # per-cycle loudness randomness
    flutter: float = 0.008     # slow natural pitch drift on held sounds
    breath: float = 0.08       # breath noise vs. buzz (loudness ratio)
    tension: float = 0.0       # -1 lax/soft/breathy .. +1 pressed/firm (moods add to this)
    roughness: float = 0.0     # 0..1 hoarseness (old, gruff)
    wobble_hz: float = 0.0     # old voice: slow steady wobble (~2 Hz), NOT random pitch
    wobble_depth: float = 0.0
    pitch_range: float = 1.0   # how much the tune moves pitch
    gain: float = 1.0
    rate: float = 1.0          # talking speed (elder slower, child quicker)


CAST: dict[str, Voice] = {v.name: v for v in [
    Voice("guard", 100, "man", tube_scale=0.96, tension=0.25, breath=0.05, pitch_range=0.85),
    Voice("lady", 215, "woman", tension=-0.35, breath=0.2, pitch_range=1.35, rate=1.05,
          jitter=0.005),
    Voice("child", 290, "child", tension=-0.15, breath=0.12, pitch_range=1.5, rate=1.12),
    Voice("pirate", 105, "man", mouth_closed=0.8, tension=0.3, roughness=0.12, breath=0.14,
          jitter=0.01, gain=1.1),
    Voice("elder", 128, "man", tube_scale=0.98, tension=-0.1, roughness=0.18, breath=0.25,
          jitter=0.018, shimmer=0.09, wobble_hz=2.0, wobble_depth=0.04, pitch_range=0.65,
          rate=0.82),
    Voice("granny", 175, "woman", tube_scale=0.97, tension=-0.3, roughness=0.12, breath=0.3,
          jitter=0.016, shimmer=0.08, wobble_hz=2.2, wobble_depth=0.045, pitch_range=0.8,
          rate=0.82),
    Voice("cat", 420, "child", tube_scale=1.15, tension=-0.1, pitch_range=1.3),
]}


def get_voice(name: str) -> Voice:
    try:
        return CAST[name]
    except KeyError:
        raise KeyError(f"unknown voice {name!r}; choose from {', '.join(CAST)}") from None
