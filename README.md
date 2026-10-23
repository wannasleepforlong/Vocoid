# Vocoid

Mumble speech for game dialogue: NPCs "talk" in synthesized babble while their text types out.
No recordings. Every syllable is built live from math (glottal buzz + vocal-tract formants).

- Real English pronunciation (CMU dictionary) picks vowels, syllables and stress; made-up names use letter rules
- Speech-like rhythm: pauses only at punctuation, words in a phrase run together
- Human pitch: smooth glides, natural sag, jitter/shimmer, tune from punctuation (`?` rises, `!` starts high, `...` fades, CAPS shouts)
- Voices: guard, lady, child, pirate, elder, granny, cat. Each uses measured vowel tables for
  men, women or children (Peterson & Barney), plus its own voice quality: firm vs. breathy,
  hoarse roughness and wobble for old voices
- Emotions: neutral, happy, sad, angry, scared, surprised, tired. Each changes pitch level and range,
  speed, loudness, breathiness, tremor and voice tension (soft/dark vs. pressed/bright)
- Inline tags: `{mood:sad}`, `{mood:happy:0.5}` (strength), switchable mid-line; sound effects at an
  exact letter: `Mrrow.{purr} Hello!`

## Setup

```bash
pip install -r requirements.txt
```

## Use

```bash
python demo.py                                                   # live text box
python -m vocoid.render "Is that you?" --voice elder -o out.wav  # one line -> WAV
python -m vocoid.render "Hello there, friend." --cast -o cast.wav # every voice
python -m vocoid.render "I lost it all." --voice lady --mood sad -o sad.wav
python -m vocoid.render "You did WHAT?" --moods -o moods.wav      # every mood back to back
python -m vocoid.render "{mood:happy}Great news! {mood:sad}But the cat left." -o mix.wav
python -m pytest -q
```

Demo keys: Space skip/next, 1-6 voice, E mood, R replay, M mode (voice / beep baselines), `-` / `=` speed.

See [PLAN.md](PLAN.md) for the design and roadmap.
