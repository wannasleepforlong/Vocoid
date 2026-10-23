import numpy as np
import pytest

from vocoid import SR
from vocoid.emotion import EMOTIONS, NEUTRAL, get_emotion
from vocoid.engine import Typewriter, VoiceHit
from vocoid.render import render_line
from vocoid.synth.syllable import synth_syllable
from vocoid.text.markup import parse_tags
from vocoid.voice import CAST, Voice

LADY = CAST["lady"]
LINE = "The road ahead is long, but the stars are bright tonight."


def hits(line, mood=None, voice=LADY):
    return [e for e in Typewriter(line, voice, mood=mood).run_all() if isinstance(e, VoiceHit)]


def semitones(hs):
    return 12 * np.log2(np.array([h.ratio for h in hs]))


def test_mood_tags_parse():
    m = parse_tags("{mood:sad}I lost it.{purr} {mood:happy:0.5}Oh well!")
    assert m.text == "I lost it. Oh well!"
    assert m.moods == {0: ("sad", 1.0), 11: ("happy", 0.5)}
    assert m.sfx == {9: ["purr"]}


def test_unknown_mood_errors():
    with pytest.raises(KeyError):
        Typewriter("{mood:hangry}hi", LADY)


def test_blend():
    assert get_emotion("sad", 0.0) == get_emotion(NEUTRAL).blend(1) or get_emotion("sad", 0.0).pitch == 1.0
    half = get_emotion("sad", 0.5)
    assert half.pitch == pytest.approx(0.95) and half.rate == pytest.approx(0.89)


def test_neutral_is_not_monotone():
    st = semitones(hits(LINE))
    assert st.max() - st.min() > 4.0          # real melody across the line
    assert np.mean(np.abs(np.diff(st))) > 0.8  # syllable-to-syllable movement


def test_happy_vs_sad():
    happy, sad = hits(LINE, "happy"), hits(LINE, "sad")
    sh, ss = semitones(happy), semitones(sad)
    assert sh.mean() > ss.mean() + 3                       # higher
    assert (sh.max() - sh.min()) > 2 * (ss.max() - ss.min())  # livelier
    assert sad[-1].time > happy[-1].time * 1.25            # slower
    assert np.mean([h.gain for h in sad]) < np.mean([h.gain for h in happy])


def test_angry_is_loud_and_punchy():
    angry, neutral = hits(LINE, "angry"), hits(LINE)
    assert np.mean([h.gain for h in angry]) > 1.25 * np.mean([h.gain for h in neutral])


def test_mid_line_switch():
    hs = hits("{mood:happy}Great news, friend. {mood:sad}But the cat left.")
    names = [h.emotion.name for h in hs]
    assert names[0] == "happy" and names[-1] == "sad"
    assert names == sorted(names, key=lambda n: n != "happy")  # happy first, then sad


def test_every_mood_renders():
    for name in EMOTIONS:
        y = render_line("Hello there, friend!", LADY, mood=name)
        assert np.isfinite(y).all() and 0.05 < np.max(np.abs(y)) <= 0.98


def _brightness(y):
    """High band (2-8 kHz) energy vs. the F1/F2 region."""
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    return spec[f > 2000].sum() / spec[(f > 300) & (f < 1500)].sum()


def test_tension_changes_voice_quality():
    clean = Voice("clean", 120, jitter=0, shimmer=0, flutter=0, breath=0)
    kw = dict(steady=True, envelope=False, rng=np.random.default_rng(0))
    angry = synth_syllable("ah", clean, 1.0, 0.5, emotion=EMOTIONS["angry"], **kw)
    sad = synth_syllable("ah", clean, 1.0, 0.5, emotion=EMOTIONS["sad"], **kw)
    assert _brightness(angry) > 1.5 * _brightness(sad)  # pressed = brighter, lax = darker


def test_scared_trembles():
    from vocoid.synth.syllable import pitch_curve
    f0 = pitch_curve(LADY, 1.0, SR, SR, steady=True, emotion=EMOTIONS["scared"])
    assert f0.max() / f0.min() > 1.04
