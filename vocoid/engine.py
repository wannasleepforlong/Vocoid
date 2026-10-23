"""Typewriter: letters appear on a speech-rhythm timeline; a syllable's letter triggers its sound.

Sound still hangs off the letter appearing, so the voice can never drift from the text. Pure logic,
no audio: the live player and the offline renderer both consume the same events.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from vocoid.emotion import NEUTRAL, Emotion, get_emotion
from vocoid.text.markup import parse_tags
from vocoid.text.prosody import plan_line, split_phrases
from vocoid.text.rhythm import timeline
from vocoid.text.syllables import find_syllables
from vocoid.voice import Voice


@dataclass(frozen=True)
class CharRevealed:
    time: float
    index: int
    char: str


@dataclass(frozen=True)
class VoiceHit:
    time: float
    index: int
    vowel: str
    ratio: float               # pitch vs. the voice's base pitch
    gain: float
    duration: float            # seconds
    glide_from: float | None   # previous syllable's pitch in the same phrase, else None
    vowel_end: str | None = None  # diphthong target ("right" = ah -> ee)
    emotion: Emotion = NEUTRAL
    phrase: int = 0               # syllables of one phrase are rendered as one continuous voice


@dataclass(frozen=True)
class SfxHit:
    time: float
    index: int
    name: str


Event = CharRevealed | VoiceHit | SfxHit


class Typewriter:
    def __init__(self, line: str, voice: Voice, speed: float = 1.0,
                 mood: str | Emotion | None = None, mood_amount: float = 1.0):
        """mood = default mood for the line; {mood:x} tags in the line override it from that point."""
        tags = parse_tags(line)
        self.text, self.sfx = tags.text, tags.sfx
        self.voice = voice
        self.speed = speed * voice.rate
        self._mood_marks = sorted((i, get_emotion(name, amt)) for i, (name, amt) in tags.moods.items())
        self.default_mood = get_emotion(mood, mood_amount)
        self.revealed = 0
        self.clock = 0.0
        self.hits, self.char_times = self._schedule()

    def mood_at(self, index: int) -> Emotion:
        mood = self.default_mood
        for i, e in self._mood_marks:
            if i > index:
                break
            mood = e
        return mood

    def _schedule(self) -> tuple[dict[int, VoiceHit], list[float]]:
        syllables = find_syllables(self.text)
        notes = plan_line(self.text, syllables, self.mood_at)
        by_index = {s.index: s for s in syllables}
        sounds, char_times = timeline(self.text, syllables, notes, split_phrases(self.text),
                                      self.speed, self.mood_at)
        v = self.voice
        hits: dict[int, VoiceHit] = {}
        prev = None
        for i in sorted(sounds):
            note, snd, mood = notes[i], sounds[i], self.mood_at(i)
            ratio = max(0.5, mood.pitch * (1.0 + note.pitch * v.pitch_range * mood.range))
            # pitch flows on within a phrase; after punctuation it starts fresh
            glide = prev[1] if prev is not None and prev[0] == note.phrase else None
            hits[i] = VoiceHit(snd.time, i, note.vowel, ratio, note.gain * v.gain * mood.gain,
                               snd.duration, glide, by_index[i].vowel_end, mood, note.phrase)
            prev = (note.phrase, ratio)
        return hits, char_times

    def phrase_hits(self, phrase: int) -> list[VoiceHit]:
        return sorted((h for h in self.hits.values() if h.phrase == phrase), key=lambda h: h.time)

    @property
    def phrases(self) -> list[int]:
        return sorted({h.phrase for h in self.hits.values()})

    @property
    def done(self) -> bool:
        return self.revealed >= len(self.text)

    @property
    def visible(self) -> str:
        return self.text[:self.revealed]

    @property
    def length(self) -> float:
        return self.char_times[-1] if self.char_times else 0.0

    def update(self, dt: float) -> list[Event]:
        """Advance the clock by dt seconds; return everything that happened, with exact times."""
        self.clock += dt
        events: list[Event] = []
        while not self.done and self.char_times[self.revealed] <= self.clock:
            i = self.revealed
            t = self.char_times[i]
            self.revealed += 1
            events.append(CharRevealed(t, i, self.text[i]))
            if i in self.hits:
                events.append(self.hits[i])
            for name in self.sfx.get(i, ()):
                events.append(SfxHit(t, i, name))
        return events

    def skip(self) -> None:
        """Show the whole line now. No letters left to type, so the voice just stops."""
        self.revealed = len(self.text)

    def run_all(self) -> list[Event]:
        return self.update(math.inf)
