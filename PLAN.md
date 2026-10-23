# Vocoid — Mumble Speech in Python (Plan)

Goal: rebuild the video's "mumble speech" system. Text types out letter by letter; a synthetic,
formant-based voice babbles along with it. No recordings. Per-character voices. Tune comes from punctuation.

## Stack

- Python 3.11+
- `numpy`, `scipy` — synthesis + filters
- `pygame` — text-box demo window + audio mixing (many overlapping channels)
- `scipy.io.wavfile` — bake / export WAVs
- `pytest` — tests

## Layout

```
vocoid/
  synth/
    glottal.py      # buzz source: pulse train + jitter + breath noise
    formants.py     # vowel formant table + resonator filters + tube scaling
    envelope.py     # swell / hold / release envelope
    bake.py         # render vowel bank per voice -> WAV / numpy cache
  voice.py          # Voice dataclass + cast presets
  text/
    timing.py       # per-char delay rules
    syllables.py    # vowel-group detection + letter -> vowel mapping
    prosody.py      # phrase split, pitch drift, ? ! ... CAPS rules
    markup.py       # inline sound-effect tags, e.g. "meow{purr} meow"
  engine.py         # Typewriter + Mumbler: timer -> letter -> sound event
  player.py         # pygame playback (pitch shift by resample, overlap mixing)
  render.py         # offline: line + voice -> one WAV (for fast tuning, no UI)
demo.py             # pygame text box, cast picker, skip button
tests/
```

## Build order (follows the video, each step listenable)

### 1. Typewriter timer (`text/timing.py`, `engine.py`)
- One timer. Fires -> reveal next char -> restart timer with delay picked by the char just typed.
- Normal letter = base (configurable, ~1/30 s). Comma = 3x. `. ? !` = 6x (2x comma). Space = normal.
- Engine emits events: `CharRevealed(index, char)`. Sound hangs off this event, so voice never drifts from text.
- Skip: reveal all at once, emit no more events -> voice stops naturally.

### 2. Baseline beep (throwaway)
- Sine beep per letter. Then random pitch per beep. Confirms "buzz" + "chaos" problems. Keep behind a flag for comparison.

### 3. Formant vowel synth (`synth/`)
- Source: glottal pulse train at f0 (Rosenberg / sawtooth-ish pulse, not pure sine).
- Filter: 3 parallel or cascaded 2-pole resonators (Klatt style) at F1/F2/F3 with bandwidths.
- 6 vowels — ah, eh, ee, oh, ooh, uh — formants from Peterson & Barney (1952) male averages.
- Fixes to go from "steel drum" to "voice":
  1. Envelope: swell in (~25 ms), hold, release (~40 ms). No hit-then-decay.
  2. Breath: mix low-level filtered noise (aspiration), ~5–10%.
  3. Jitter: each glottal period length + amplitude randomly varied ~1–2%.
- Sound length ~120–150 ms (a bit longer than gap between syllables -> overlap -> continuous babble).

### 4. Voices (`voice.py`)
```python
@dataclass
class Voice:
    name: str
    base_pitch: float      # Hz of buzz
    tube_scale: float      # multiplies F1-F3 (shorter tube -> higher)
    mouth_closed: float    # 0..1, lowers F1/F2, narrows (pirate mutter)
    jitter: float
    breath: float
    wobble_hz: float = 0   # old voice: ~2 Hz steady vibrato
    wobble_depth: float = 0
    pitch_range: float = 1 # how much prosody moves pitch
```
- Cast: guard (low pitch, scale 1.0), woman (scale ~1.17, higher pitch), child (scale ~1.3+), pirate (mouth closed),
  old (slow 2 Hz wobble, NOT random pitch), small creature (high pitch).
- Key lesson baked in: pitch alone = "same guy straining"; also scale the tube.
- Bake each voice's 6 vowels once at its base pitch -> cache. Runtime pitch changes = resample / playback-rate
  (small shifts only, so formant smearing stays minor). Target total bank ~0.5 MB.
- New character = one line: `Voice("Agnes", 210, 1.17, ...)`.

### 5. Syllables (`text/syllables.py`)
- Play sound only when a vowel group starts (`aeiou`, `y` when not word-initial). Others silent.
  ~30 sounds/s -> ~10/s.
- Letter -> vowel: a->ah, e->eh, i->ee, o->oh, u->uh; digraphs: `oo`/`ou`->ooh, `ee`/`ea`->ee, etc.
- Deterministic: same line -> same sounds. ("Sister Agnes" -> ee eh ah eh.)
- Unit tests on known words.

### 6. Prosody (`text/prosody.py`)
- Pre-parse the whole line before typing (needs lookahead for "last word" of a question).
- Split into phrases at `, . ? ! ; :` and `...`.
- Pitch per syllable = sum of 4 parts:
  1. voice base pitch
  2. phrase position: drifts down across phrase, resets up at next phrase / comma
  3. tiny random nudge (seeded from line hash -> repeatable)
  4. wobble if old (sine on wall-clock time)
- Punctuation rules:
  - `?` -> last word rises
  - `!` -> phrase starts high
  - `...` -> volume fades across the phrase
  - ALL CAPS word -> shout (louder, higher)
- Output: per-char-index table `{index: SyllableEvent(vowel, pitch_mult, gain)}`.

### 7. Engine glue (`engine.py`)
- On `CharRevealed(i)`: if `i` in syllable table -> pick vowel from voice bank, apply pitch + gain, play.
- Markup: `{purr}` style tags stripped from display text, stored as `{index: sfx_name}`, fired at that letter.
- (Optional, tried and dropped in video) consonant noise puffs for s/t/k — add behind flag, default off.

### 8. Playback + demo (`player.py`, `demo.py`)
- pygame mixer, ~16 channels so overlapping syllables don't cut each other.
- Pitch shift: resample numpy array (`scipy.signal.resample` or linear interp) and cache per (vowel, rounded pitch).
- Demo: text box, Space = skip/next line, number keys = switch voice, line: same sentence by 3 voices.

### 9. Offline render (`render.py`)
- `python -m vocoid.render "Sister Agnes, is that you?" --voice old -o out.wav`
- Simulates timer in sample time and mixes sounds at exact offsets. Main tool for tuning without UI.

## 10. Humanize — go past "somebody humming"

Video stops at humming. Below = extra layers, ordered cheap -> big. Each behind a flag, A/B via `render.py`.

### Tier A — cheap wins (do first)
- **Live synth per syllable** instead of baked bank. numpy renders 150 ms in well under 1 ms. Unlocks everything
  below. Also kills resample "chipmunk" smear (pitch moves, formants stay put — correct).
- **Better glottal source**: LF model (Liljencrants–Fant) instead of plain pulse. Per-voice `tension` param:
  breathy (soft, steep spectral tilt) <-> pressed (bright, harsh). Big part of "who" a voice is.
- **Shimmer** (amplitude jitter) next to jitter. Real voices have both.
- **Lip radiation**: first-difference filter (+6 dB/oct). Cheap, adds presence.
- **F4 + F5** fixed-ish formants (~3.5 / 4.5 kHz × tube_scale). Adds "realness" / timbre ring.
- **Pitch inside syllable**: small rise-fall contour across each sound, not flat tone. Flat = synth giveaway.
- **Timing jitter**: ±10–15% random on letter timer. Perfect metronome = robot.

### Tier B — coarticulation + rhythm (biggest jump)
- **Formant glides**: each syllable starts at previous vowel's formants, slides to target over ~40 ms.
  Real speech never jumps. Also gives free diphthongs: `ai` (ah->ee), `oi`, `ou` (ah->ooh), `ay`.
- **Sonorant onsets, not clicks**: video's consonants failed because noise puffs sit alone. Instead shape the
  *vowel onset* from the consonant before it:
  - `m n` -> short nasal murmur (low formant ~250 Hz + anti-formant), then open
  - `l r w y` -> start at that sound's formants, glide into vowel
  - `b d g p t k` -> 20–40 ms silence before onset + tiny burst filtered by next vowel's formants, low gain
  - `s sh f` -> soft band-limited noise, much quieter than vowel, blended in
- **Stress**: guess stressed syllable (first syllable of content words; skip "the/a/of/to/and"). Stressed =
  longer, louder, pitch accent bump. Unstressed vowels drift toward `uh` (schwa). Gives English its bounce.
- **Phrase-final lengthening**: last syllable before `, . ? !` ~1.5× longer.
- **Vowel length**: ee/ooh/ah long, eh/uh short.
- **Phrase-end voice quality**: last syllable gets creak (vocal fry: low irregular pulses) or extra breath. Very human.
- **Breath intakes**: soft inhale noise after sentence break, before next long line.

### Tier C — physical tube model (optional, most natural)
- Replace formant filters with a **Kelly–Lochbaum waveguide**: vocal tract as ~44 tube segments with
  per-segment diameter (same idea as Pink Trombone). Vowels = tract shapes; consonants = constrictions;
  glides + nasals come out naturally. Tube length = voice size, literally.
- Heavier CPU: needs per-sample loop -> `numba` JIT. Still offline-cheap per syllable.
- Keep formant engine as fallback.

### Emotion + polish
- Per-line mood tag `{mood:sad}`: changes pitch range, speed, tension, breath.
  happy = wide range + fast; sad = narrow + low + slow + breathy; angry = pressed + loud; scared = fast + jittery.
- Small room reverb (short convolution) + gentle compressor so overlapping syllables glue together.
- Short crossfades at sound edges -> no clicks.

### What NOT to chase
- Real words / TTS (e.g. Piper, Coqui). Possible, but leaves mumble style, adds big model downloads.
  Mumble should stay "almost words".

### Proof
- `render.py --compare` renders same line through baseline / Tier A / Tier B / Tier C into one WAV, back to back.
- Spectrogram dump (`matplotlib`) to check glides + formant positions by eye.

## Tests
- timing delays per punctuation
- syllable detection + vowel mapping
- prosody: question rise, caps shout, comma reset
- determinism: same line renders same bytes
- synth sanity: formant peaks near expected Hz (FFT check)

## Open choices (defaults picked)
- Runtime pitch: synth live each syllable (default now, needed for glides) vs. resample baked vowels.
- Engine: formant filters (default) vs. Kelly–Lochbaum tube (Tier C, needs `numba`).
- UI: pygame (default) vs. CLI-only WAV export.
