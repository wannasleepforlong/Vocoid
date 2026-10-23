"""A voice = buzz pitch + tube size + a few quirks. New character = one line."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Voice:
    name: str
    base_pitch: float          # Hz of the buzz
    tube_scale: float = 1.0    # >1 = shorter tube (woman, child). Pitch alone sounds like "same guy straining".
    mouth_closed: float = 0.0  # 0..1, mutter with mouth almost shut
    jitter: float = 0.006      # per-cycle pitch randomness (humans ~0.5-1%)
    shimmer: float = 0.03      # per-cycle loudness randomness
    flutter: float = 0.008     # slow natural pitch drift on held sounds
    breath: float = 0.08       # breath noise vs. buzz (loudness ratio)
    wobble_hz: float = 0.0     # old voice: slow steady wobble (~2 Hz), NOT random pitch
    wobble_depth: float = 0.0
    pitch_range: float = 1.0   # how much the tune moves pitch
    gain: float = 1.0
    rate: float = 1.0          # talking speed (elder slower, child quicker)


CAST: dict[str, Voice] = {v.name: v for v in [
    Voice("guard", 98, tube_scale=0.95, pitch_range=0.85),
    Voice("lady", 205, tube_scale=1.17, breath=0.14),
    Voice("child", 285, tube_scale=1.32, pitch_range=1.2, breath=0.1, rate=1.1),
    Voice("pirate", 108, mouth_closed=0.8, breath=0.16, jitter=0.01, gain=1.1),
    Voice("elder", 135, tube_scale=1.02, wobble_hz=2.0, wobble_depth=0.04, breath=0.18,
          jitter=0.01, shimmer=0.05, rate=0.85),
    Voice("cat", 420, tube_scale=1.5, pitch_range=1.3),
]}


def get_voice(name: str) -> Voice:
    try:
        return CAST[name]
    except KeyError:
        raise KeyError(f"unknown voice {name!r}; choose from {', '.join(CAST)}") from None
