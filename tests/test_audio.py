import numpy as np
import pytest

from vocoid import SR
from vocoid.engine import CharRevealed, SfxHit, Typewriter, VoiceHit
from vocoid.render import render_cast, render_line
from vocoid.synth.bake import get_bank, synth_vowel
from vocoid.synth.syllable import pitch_curve, synth_syllable
from vocoid.voice import CAST, Voice

GUARD = CAST["guard"]


def test_typewriter_events():
    tw = Typewriter("Hi, you.{purr}", GUARD)
    ev = tw.run_all()
    chars = [e for e in ev if isinstance(e, CharRevealed)]
    assert "".join(e.char for e in chars) == "Hi, you."
    hits = [e for e in ev if isinstance(e, VoiceHit)]
    assert [e.index for e in hits] == [1, 5]
    assert hits[0].vowel == "ah" and hits[0].vowel_end == "ee"  # "hi" slides ah -> ee
    assert [e.index for e in ev if isinstance(e, SfxHit)] == [7]
    times = [e.time for e in chars]
    assert times == sorted(times) and times[0] == 0.0
    assert times[5] - times[2] > 0.2  # comma pause before "you"


def test_typewriter_incremental_and_skip():
    tw = Typewriter("hello there friend", GUARD)
    assert len([e for e in tw.update(0.0) if isinstance(e, CharRevealed)]) == 1
    tw.update(0.1)
    assert 1 < tw.revealed < len(tw.text)
    tw.skip()
    assert tw.done and tw.visible == tw.text
    assert tw.update(10) == []


def test_old_voice_wobbles():
    flat = Voice("flat", 130)
    old = Voice("old", 130, wobble_hz=2.0, wobble_depth=0.05)
    a = pitch_curve(flat, 1.0, SR, SR, steady=True)
    b = pitch_curve(old, 1.0, SR, SR, steady=True)
    assert np.allclose(a, 130)
    assert b.max() > 135 and b.min() < 125


def test_hits_glide_and_chain():
    hits = [e for e in Typewriter("one two three, four five.", GUARD).run_all() if isinstance(e, VoiceHit)]
    # one two three | four five
    assert [h.glide_from is None for h in hits] == [True, False, False, True, False]
    for p, h in zip(hits, hits[1:]):
        if h.glide_from is not None:
            assert h.glide_from == p.ratio
    assert hits[2].duration > hits[1].duration  # phrase-final lengthening ("three" before comma)


def test_diphthong_slides():
    """'ah>ee' must end with ee's high F2, not stay on ah."""
    clean = Voice("clean", F0, jitter=0.0, shimmer=0.0, flutter=0.0, breath=0.0)
    y = synth_syllable("ah", clean, 1.0, 0.5, steady=True, envelope=False, vowel_end="ee")
    q = len(y) // 5
    assert _band_centroid(y[-q:], 1500, 3000) > _band_centroid(y[:q], 1500, 3000) + 150
    assert _band_centroid(y[-q:], 200, 1000) < _band_centroid(y[:q], 200, 1000) - 100


def f0_track(y, sr=SR, lo=60, hi=600, win=0.03, hop=0.005):
    """Autocorrelation pitch track (Hz) on voiced frames."""
    n, h, out = int(win * sr), int(hop * sr), []
    for s in range(0, len(y) - n, h):
        x = y[s:s + n] - y[s:s + n].mean()
        if np.sqrt(np.mean(x ** 2)) < 0.03:
            continue
        ac = np.correlate(x, x, "full")[n - 1:]
        a, b = int(sr / hi), int(sr / lo)
        lag = a + int(np.argmax(ac[a:b]))
        if 1 <= lag < len(ac) - 1:  # parabolic refine -> sub-sample pitch
            d = ac[lag - 1] - 2 * ac[lag] + ac[lag + 1]
            lag = lag + 0.5 * (ac[lag - 1] - ac[lag + 1]) / d if d else lag
        out.append(sr / lag)
    return np.array(out)


@pytest.mark.parametrize("name", ["guard", "lady", "child"])
def test_pitch_hits_target_and_moves(name):
    v = CAST[name]
    y = synth_syllable("ah", v, 1.0, 0.3, glide_from=1.2, rng=np.random.default_rng(1))
    track = f0_track(y, lo=v.base_pitch / 1.6, hi=v.base_pitch * 1.6)
    st = 12 * np.log2(track / v.base_pitch)
    third = len(st) // 3
    assert abs(np.median(st)) < 1.0                          # within a semitone of the target
    assert np.median(st[:third]) > np.median(st[-third:])   # glides down from higher start, then sags
    assert np.percentile(np.abs(np.diff(st)), 95) < 0.5      # smooth, no jumps


def _band_centroid(y, lo=300, hi=3000):
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    m = (f > lo) & (f < hi)
    return float((f[m] * spec[m]).sum() / spec[m].sum())


def test_higher_pitch_keeps_mouth_shape():
    """Doubling pitch must not move the mouth (old resample approach doubled it)."""
    clean = Voice("clean", F0, jitter=0.0, shimmer=0.0, flutter=0.0, breath=0.0)
    low = synth_vowel("ah", clean, SR, 1.0, f0=F0, envelope=False)
    high = synth_vowel("ah", clean, SR, 1.0, f0=2 * F0, envelope=False)
    # resampling would shift it ~100%; sparser harmonics alone move the centroid ~15-20%
    assert abs(_band_centroid(high) / _band_centroid(low) - 1) < 0.3


F0 = 105  # divides SR exactly, so harmonics sit on clean periods


def _f1_peak(vowel, voice, f0=F0):
    """Loudest harmonic below 1200 Hz (local max per harmonic, robust to bin offsets)."""
    y = synth_vowel(vowel, voice, SR, duration=1.0, f0=f0, envelope=False)
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    freqs = np.fft.rfftfreq(len(y), 1 / SR)
    harmonics = np.arange(f0, 1200, f0)
    amps = [spec[(freqs > h - 15) & (freqs < h + 15)].max() for h in harmonics]
    return harmonics[int(np.argmax(amps))]


def test_formant_peaks():
    clean = Voice("clean", F0, jitter=0.0, breath=0.0)
    assert 600 <= _f1_peak("ah", clean) <= 850
    assert 150 <= _f1_peak("ee", clean) <= 400


def test_shorter_tube_raises_formants():
    clean = Voice("clean", F0, jitter=0.0, breath=0.0)
    short = Voice("short", F0, tube_scale=1.3, jitter=0.0, breath=0.0)
    assert _f1_peak("ah", short) > _f1_peak("ah", clean)


def test_bank():
    bank = get_bank(GUARD)
    assert set(bank) == {"ah", "ae", "eh", "ih", "ee", "aw", "oh", "uu", "ooh", "uh", "er"}
    assert all(np.max(np.abs(y)) < 0.9 for y in bank.values())


def test_bank_size_small():
    total = sum(len(v) * 2 for voice in CAST.values() for v in get_bank(voice).values())
    assert total / len(CAST) < 130_000  # export-only bank (playback is live): ~120 KB per voice


def test_render_deterministic_and_audible():
    a = render_line("Sister Agnes, is that you?", CAST["elder"])
    b = render_line("Sister Agnes, is that you?", CAST["elder"])
    assert np.array_equal(a, b)
    assert np.max(np.abs(a)) > 0.1 and np.max(np.abs(a)) <= 0.98


def test_render_modes_and_cast():
    for mode in ("beep", "random-beep"):
        assert np.max(np.abs(render_line("hi there", GUARD, mode=mode))) > 0.1
    assert len(render_cast("hi", list(CAST.values()))) > len(render_line("hi", GUARD)) * 3
