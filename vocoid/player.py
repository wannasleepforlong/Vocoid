"""Live playback through pygame's mixer.

Each phrase is one continuous sound. It is rendered when the line starts (prepare) and starts
playing the moment its first syllable's letter appears, so voice and text stay locked together.
"""

from __future__ import annotations

import numpy as np
import pygame

from vocoid import SR
from vocoid.engine import CharRevealed, Event, SfxHit, Typewriter, VoiceHit
from vocoid.render import beep
from vocoid.synth.bake import to_int16
from vocoid.synth.phrase import synth_phrase
from vocoid.synth.sfx import SFX, get_sfx

HEADROOM = 0.65  # loud notes (shouts) can go above 1.0 gain


class Player:
    def __init__(self, sr: int = SR, channels: int = 32):
        if not pygame.mixer.get_init():
            pygame.mixer.pre_init(sr, -16, 1, 512)
            pygame.mixer.init()
        self.sr, _, self.out_channels = pygame.mixer.get_init()
        pygame.mixer.set_num_channels(channels)
        self._cache: dict[tuple, pygame.mixer.Sound] = {}
        self._phrases: dict[int, pygame.mixer.Sound] = {}
        self._tw: Typewriter | None = None
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

    def prepare(self, tw: Typewriter) -> None:
        """Render every phrase of a new line up front (stops whatever was playing)."""
        self.stop()
        self._tw = tw
        self._phrases = {}
        for p in tw.phrases:
            _, y = synth_phrase(tw.phrase_hits(p), tw.voice, tw.text, self.sr)
            self._phrases[p] = self._sound(y)

    def handle(self, events: list[Event], tw: Typewriter, mode: str = "voice") -> None:
        if tw is not self._tw:
            self.prepare(tw)
        for e in events:
            if isinstance(e, VoiceHit) and mode == "voice":
                if e.glide_from is None and e.phrase in self._phrases:  # phrase's first syllable
                    self._phrases[e.phrase].play()
            elif isinstance(e, CharRevealed) and mode != "voice" and e.char.isalnum():
                freq = 520 if mode == "beep" else int(self._rng.uniform(250, 950))
                self._play_cached(("beep", freq), lambda: beep(freq, self.sr))
            elif isinstance(e, SfxHit) and e.name in SFX:
                self._play_cached(("sfx", e.name), lambda: get_sfx(e.name, self.sr))

    def stop(self) -> None:
        """Silence everything (used on skip: no letters left, so the voice stops)."""
        pygame.mixer.stop()
