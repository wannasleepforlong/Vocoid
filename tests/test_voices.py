import numpy as np
import pytest

from vocoid import SR
from vocoid.render import render_line
from vocoid.synth.formants import KINDS, VOWELS_BY_KIND, formants_for
from vocoid.synth.glottal import buzz
from vocoid.synth.syllable import synth_syllable
from vocoid.voice import CAST, Voice

LINE = "Sister Agnes, is that you? I thought you were lost at sea."


def band_db(y, lo, hi, ref=(500, 1000)):
    s = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    r = s[(f >= ref[0]) & (f < ref[1])].sum()
    return 10 * np.log10(s[(f >= lo) & (f < hi)].sum() / r)


def test_tables_cover_same_vowels():
    names = set(VOWELS_BY_KIND["man"])
    assert all(set(VOWELS_BY_KIND[k]) == names for k in KINDS)


def test_woman_is_not_uniform_scaling():
    """Measured tables: 'cat' F1 rises much more than 'father' F1 from man to woman."""
    ae = formants_for("ae", kind="woman")[0][0] / formants_for("ae", kind="man")[0][0]
    ah = formants_for("ah", kind="woman")[0][0] / formants_for("ah", kind="man")[0][0]
    assert ae > ah + 0.1


def test_presence_restores_upper_formants():
    """Speech-like balance: 2-4 kHz should sit within ~25 dB of the F1 band (was ~-35)."""
    v = Voice("clean", 120, jitter=0, shimmer=0, flutter=0, breath=0)
    y = synth_syllable("ah", v, 1.0, 1.0, steady=True, envelope=False)
    assert band_db(y, 2000, 4000) > -25


def test_cast_voices_are_spread_apart():
    """Brightness (spectral centroid) must separate man < woman < child."""
    cent = {}
    for name in ("guard", "lady", "child"):
        y = render_line(LINE, CAST[name])
        s = np.abs(np.fft.rfft(y)) ** 2
        f = np.fft.rfftfreq(len(y), 1 / SR)
        cent[name] = np.sum(f * s) / np.sum(s)
    assert cent["guard"] * 1.25 < cent["lady"] < cent["child"]


def test_roughness_adds_subharmonic():
    """Hoarse voice: energy appears an octave below the pitch (alternating cycles)."""
    def sub(r):
        src, _ = buzz(150, SR, 0, 0, 0, np.random.default_rng(0), duration=1.0, roughness=r)
        s = np.abs(np.fft.rfft(src))
        f = np.fft.rfftfreq(len(src), 1 / SR)
        return s[(f > 70) & (f < 80)].max() / s[(f > 145) & (f < 155)].max()
    assert sub(0.18) > 5 * sub(0.0)


@pytest.mark.parametrize("name", list(CAST))
def test_every_voice_renders(name):
    y = render_line("Hello there, friend!", CAST[name])
    assert np.isfinite(y).all() and 0.05 < np.max(np.abs(y)) <= 0.98
