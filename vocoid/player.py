"""Live playback through pygame's mixer. Many channels so overlapping syllables don't cut each other."""

from __future__ import annotations

import numpy as np
import pygame

from vocoid import SR
from vocoid.engine import CharRevealed, Event, SfxHit, VoiceHit
from vocoid.render import beep
from vocoid.synth.bake import to_int16
from vocoid.synth.sfx import SFX, get_sfx
from vocoid.synth.syllable import hit_audio
from vocoid.voice import Voice

HEADROOM = 0.65  # loud notes (shouts) can go above 1.0 gain


class Player:
    def __init__(self, sr: int = SR, channels: int = 32):
        if not pygame.mixer.get_init():
            pygame.mixer.pre_init(sr, -16, 1, 512)
            pygame.mixer.init()
        self.sr, _, self.out_channels = pygame.mixer.get_init()
        pygame.mixer.set_num_channels(channels)
        self._cache: dict[tuple, pygame.mixer.Sound] = {}
        self._rng = np.random.default_rng(0)

    def _sound(self, samples: np.ndarray) -> pygame.mixer.Sound:
        data = to_int16(samples * HEADROOM)
        if self.out_channels == 2:
            data = np.column_stack([data, data])
        return pygame.mixer.Sound(buffer=np.ascontiguousarray(data).tobytes())

    def _play_cached(self, key: tuple, make) -> None:
        snd = self._cache.get(key)
        if snd is None:
            snd = self._cache[key] = self._sound(make())
        snd.play()

    def handle(self, events: list[Event], voice: Voice, line: str, mode: str = "voice") -> None:
        """line = the typewriter's display text (seeds each syllable so a line always sounds the same)."""
        for e in events:
            if isinstance(e, VoiceHit) and mode == "voice":
                # synthesized fresh each time: pitch glides and never repeats a recording
                self._sound(hit_audio(e, voice, line, self.sr)).play()
            elif isinstance(e, CharRevealed) and mode != "voice" and e.char.isalnum():
                freq = 520 if mode == "beep" else int(self._rng.uniform(250, 950))
                self._play_cached(("beep", freq), lambda: beep(freq, self.sr))
            elif isinstance(e, SfxHit) and e.name in SFX:
                self._play_cached(("sfx", e.name), lambda: get_sfx(e.name, self.sr))

    def stop(self) -> None:
        pygame.mixer.stop()
