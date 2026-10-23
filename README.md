# Vocoid

Mumble speech for game dialogue: NPCs "talk" in synthesized babble while their text types out.
No recordings. Every syllable is built live from math (glottal buzz + vocal-tract formants).

- Real English pronunciation (CMU dictionary) picks vowels, syllables and stress; made-up names use letter rules
- Speech-like rhythm: pauses only at punctuation, words in a phrase run together
- Human pitch: smooth glides, natural sag, jitter/shimmer, tune from punctuation (`?` rises, `!` starts high, `...` fades, CAPS shouts)
- Voices = pitch + tube length + quirks: guard, lady, child, pirate, elder, cat
- Inline sound effects at an exact letter: `Mrrow.{purr} Hello!`

## Setup

```bash
pip install -r requirements.txt
```

## Use

```bash
python demo.py                                                   # live text box
python -m vocoid.render "Is that you?" --voice elder -o out.wav  # one line -> WAV
python -m vocoid.render "Hello there, friend." --cast -o cast.wav # every voice
python -m pytest -q
```

Demo keys: Space skip/next, 1-6 voice, R replay, M mode (voice / beep baselines), `-` / `=` speed.

See [PLAN.md](PLAN.md) for the design and roadmap.
