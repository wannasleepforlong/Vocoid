"""Text-box demo with live mumble speech.

    python demo.py

Space / Enter : skip typing, or go to next line
1-6           : replay current line with another voice
R             : replay
- / =         : slower / faster
E             : cycle mood (script / neutral / happy / sad / angry / scared / surprised / tired)
M             : switch mode (voice / beep / random-beep) to hear the baselines
Esc           : quit
"""

from __future__ import annotations

import argparse
import os
import re
import sys

SCRIPT = [
    ("guard", "{mood:angry}Halt! Who goes there? State your business, traveler."),
    ("lady", "{mood:happy}Oh, you must be the new courier! Sister Agnes has been waiting for you."),
    ("child", "{mood:surprised}Are you a REAL adventurer? Can I see your sword?"),
    ("pirate", "{mood:tired}Arr, the sea be calm tonight... too calm, if ye ask me."),
    ("elder", "{mood:sad}In my day, we walked to the temple. {mood:happy:0.6}But oh, what a view."),
    ("lady", "{mood:scared}Did you hear that? Something is out there, in the dark."),
    ("cat", "Mrrow.{purr} Hello, human. {mood:angry:0.7}Feed me, or else!"),
    ("guard", "Welcome back, friend. The road ahead is long, but the stars are bright."),
    ("lady", "Welcome back, friend. The road ahead is long, but the stars are bright."),
    ("elder", "Welcome back, friend. The road ahead is long, but the stars are bright."),
]

W, H = 900, 320
BOX = (30, 140, W - 60, 150)
MODES = ("voice", "beep", "random-beep")


def wrap_spans(text: str, font, width: int) -> list[tuple[int, int]]:
    """Wrap the full line once, so words never jump lines while typing."""
    spans, start = [], 0
    for m in re.finditer(r"\S+\s*", text):
        if m.start() > start and font.size(text[start:m.end()].rstrip())[0] > width:
            spans.append((start, m.start()))
            start = m.start()
    spans.append((start, len(text)))
    return spans


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=float, default=0, help="headless run for N seconds, then exit")
    args = ap.parse_args(argv)
    if args.smoke:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame

    from vocoid.emotion import EMOTIONS
    from vocoid.engine import Typewriter, VoiceHit
    from vocoid.player import Player
    from vocoid.voice import CAST

    pygame.init()
    player = Player()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Vocoid - mumble speech")
    font = pygame.font.SysFont("consolas,menlo,monospace", 24)
    small = pygame.font.SysFont("consolas,menlo,monospace", 16)
    clock = pygame.time.Clock()
    voices = list(CAST.values())

    line_i, mode_i = 0, 0
    speed = 1.0
    moods = ["script"] + list(EMOTIONS)
    mood_i = 0
    voice = CAST[SCRIPT[0][0]]

    def start(v):
        nonlocal voice
        voice = v
        line = SCRIPT[line_i][1]
        if mood_i:  # override: drop the script's mood tags, use the chosen mood
            line = re.sub(r"\{mood:[^}]*\}", "", line)
            return Typewriter(line, v, speed, moods[mood_i])
        return Typewriter(line, v, speed)

    tw = start(voice)
    spans = wrap_spans(tw.text, font, BOX[2] - 40)
    elapsed, hits = 0.0, 0
    running = True
    while running:
        dt = clock.tick(120) / 1000.0
        elapsed += dt
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key in (pygame.K_SPACE, pygame.K_RETURN):
                    if not tw.done:
                        tw.skip()
                    else:
                        line_i = (line_i + 1) % len(SCRIPT)
                        tw = start(CAST[SCRIPT[line_i][0]])
                        spans = wrap_spans(tw.text, font, BOX[2] - 40)
                elif ev.key == pygame.K_r:
                    tw = start(voice)
                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    faster = ev.key in (pygame.K_EQUALS, pygame.K_KP_PLUS)
                    speed = min(3.0, speed * 1.15) if faster else max(0.3, speed / 1.15)
                    tw = start(voice)
                elif ev.key == pygame.K_e:
                    mood_i = (mood_i + 1) % len(moods)
                    tw = start(voice)
                elif ev.key == pygame.K_m:
                    mode_i = (mode_i + 1) % len(MODES)
                    tw = start(voice)
                elif pygame.K_1 <= ev.key <= pygame.K_9 and ev.key - pygame.K_1 < len(voices):
                    tw = start(voices[ev.key - pygame.K_1])

        events = tw.update(dt)
        hits += sum(isinstance(e, VoiceHit) for e in events)
        player.handle(events, voice, tw.text, MODES[mode_i])

        screen.fill((18, 16, 28))
        pygame.draw.rect(screen, (40, 36, 62), BOX, border_radius=10)
        pygame.draw.rect(screen, (150, 140, 200), BOX, 2, border_radius=10)
        screen.blit(font.render(voice.name.title(), True, (255, 210, 120)), (BOX[0] + 16, BOX[1] - 34))
        n = tw.revealed
        for row, (s, e) in enumerate(spans):
            if s >= n:
                break
            img = font.render(tw.text[s:min(e, n)], True, (235, 232, 245))
            screen.blit(img, (BOX[0] + 20, BOX[1] + 18 + row * 32))
        if tw.done and int(elapsed * 2) % 2:
            screen.blit(font.render(">", True, (255, 210, 120)), (BOX[0] + BOX[2] - 34, BOX[1] + BOX[3] - 38))
        keys = "  ".join(f"{i + 1}:{v.name}" for i, v in enumerate(voices))
        hint = f"Space next/skip   R replay   M mode [{MODES[mode_i]}]   {keys}"
        screen.blit(small.render(hint, True, (140, 135, 170)), (30, 20))
        status = (f"line {line_i + 1}/{len(SCRIPT)}   speed x{speed:.2f}  (- slower, = faster)"
                  f"   E mood [{moods[mood_i]}]")
        screen.blit(small.render(status, True, (140, 135, 170)), (30, 44))
        pygame.display.flip()

        if args.smoke and elapsed >= args.smoke:
            running = False

    pygame.quit()
    if args.smoke:
        print(f"smoke ok: {hits} voice hits in {elapsed:.1f}s, revealed {tw.revealed} chars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
