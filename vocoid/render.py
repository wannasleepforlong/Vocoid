"""Offline: line + voice -> WAV. Fast way to tune voices without the UI.

    python -m vocoid.render "Sister Agnes, is that you?" --voice elder -o out.wav
    python -m vocoid.render "Hello there, friend." --cast -o cast.wav
    python -m vocoid.render "Hello there." --mode random-beep -o chaos.wav
    python -m vocoid.render "I lost it all." --voice lady --mood sad -o sad.wav
    python -m vocoid.render "You did WHAT?" --moods -o moods.wav     # every mood back to back
    python -m vocoid.render "{mood:happy}Great news! {mood:sad}But the cat left." -o mix.wav
    python -m vocoid.render --export-bank banks/
"""

from __future__ import annotations

import argparse
import sys
import zlib

import numpy as np
from scipy.io import wavfile

from vocoid import SR
from vocoid.emotion import EMOTIONS
from vocoid.engine import CharRevealed, SfxHit, Typewriter, VoiceHit
from vocoid.synth.bake import save_bank, to_int16
from vocoid.synth.envelope import swell
from vocoid.synth.sfx import SFX, get_sfx
from vocoid.synth.syllable import hit_audio
from vocoid.voice import CAST, Voice, get_voice

MODES = ("voice", "beep", "random-beep")


def beep(freq: float, sr: int = SR, duration: float = 0.04) -> np.ndarray:
    n = int(duration * sr)
    t = np.arange(n) / sr
    return (0.25 * np.sin(2 * np.pi * freq * t) * swell(n, sr, 0.003, 0.01)).astype(np.float32)


def render_line(line: str, voice: Voice, sr: int = SR, speed: float = 1.0,
                mode: str = "voice", tail: float = 0.4, mood: str | None = None,
                mood_amount: float = 1.0) -> np.ndarray:
    """Simulate the typewriter and mix every sound at its exact sample offset."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    tw = Typewriter(line, voice, speed, mood, mood_amount)
    events = tw.run_all()
    rng = np.random.default_rng(zlib.crc32(line.encode()))
    clips: list[tuple[int, np.ndarray]] = []
    for e in events:
        start = int(round(e.time * sr))
        if isinstance(e, CharRevealed) and mode != "voice" and e.char.isalnum():
            freq = 520.0 if mode == "beep" else rng.uniform(250, 950)
            clips.append((start, beep(freq, sr)))
        elif isinstance(e, VoiceHit) and mode == "voice":
            clips.append((start, hit_audio(e, voice, tw.text, sr)))
        elif isinstance(e, SfxHit):
            if e.name in SFX:
                clips.append((start, get_sfx(e.name, sr)))
            else:
                print(f"warning: unknown sfx {{{e.name}}}", file=sys.stderr)
    end = max((s + len(c) for s, c in clips), default=0)
    out = np.zeros(end + int(tail * sr), dtype=np.float32)
    for s, c in clips:
        out[s:s + len(c)] += c
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > 0.98:
        out *= 0.98 / peak
    return out


def render_cast(line: str, voices: list[Voice], sr: int = SR, gap: float = 0.35, **kw) -> np.ndarray:
    silence = np.zeros(int(gap * sr), dtype=np.float32)
    parts = []
    for v in voices:
        parts += [render_line(line, v, sr, **kw), silence]
    return np.concatenate(parts)


def render_moods(line: str, voice: Voice, sr: int = SR, gap: float = 0.5, **kw) -> np.ndarray:
    silence = np.zeros(int(gap * sr), dtype=np.float32)
    parts = []
    for name in EMOTIONS:
        parts += [render_line(line, voice, sr, mood=name, **kw), silence]
    return np.concatenate(parts)


def write_wav(path: str, samples: np.ndarray, sr: int = SR) -> None:
    wavfile.write(path, sr, to_int16(samples))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render mumble speech to a WAV file.")
    ap.add_argument("text", nargs="?", help="line to speak; {purr} style tags fire sound effects")
    ap.add_argument("--voice", default="guard", help=f"one of: {', '.join(CAST)}")
    ap.add_argument("--cast", action="store_true", help="same line through every voice, back to back")
    ap.add_argument("--mode", choices=MODES, default="voice", help="beep modes = the naive baselines")
    ap.add_argument("--speed", type=float, default=1.0, help="talking speed, 1.0 = normal")
    ap.add_argument("--mood", choices=list(EMOTIONS), help="mood for the whole line ({mood:x} tags still override)")
    ap.add_argument("--mood-amount", type=float, default=1.0, help="mood strength 0..1")
    ap.add_argument("--moods", action="store_true", help=f"same line in every mood: {', '.join(EMOTIONS)}")
    ap.add_argument("-o", "--out", default="out.wav")
    ap.add_argument("--export-bank", metavar="DIR", help="write every voice's vowel WAVs to DIR")
    args = ap.parse_args(argv)

    if args.export_bank:
        total = sum(save_bank(v, args.export_bank) for v in CAST.values())
        print(f"wrote {len(CAST)} voices to {args.export_bank} ({total / 1024:.0f} KB of audio)")
        if not args.text:
            return 0
    if not args.text:
        ap.error("text is required")

    kw = dict(speed=args.speed, mode=args.mode)
    if args.moods:
        samples = render_moods(args.text, get_voice(args.voice), mood_amount=args.mood_amount, **kw)
    elif args.cast:
        samples = render_cast(args.text, list(CAST.values()), mood=args.mood,
                              mood_amount=args.mood_amount, **kw)
    else:
        samples = render_line(args.text, get_voice(args.voice), mood=args.mood,
                              mood_amount=args.mood_amount, **kw)
    write_wav(args.out, samples)
    print(f"wrote {args.out} ({len(samples) / SR:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
