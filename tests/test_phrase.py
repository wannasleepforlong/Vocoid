import numpy as np

from vocoid import SR
from vocoid.engine import Typewriter, VoiceHit
from vocoid.render import render_line
from vocoid.synth.phrase import group_phrases, synth_phrase
from vocoid.voice import CAST

ELDER = CAST["elder"]


def test_one_sound_per_phrase():
    tw = Typewriter("One two three, four five.", ELDER)
    hits = [e for e in tw.run_all() if isinstance(e, VoiceHit)]
    groups = group_phrases(hits)
    assert [len(g) for g in groups] == [3, 2]
    assert tw.phrases == [0, 1]
    assert all(h.glide_from is None for h in (g[0] for g in groups))


def test_phrase_is_continuous_and_starts_on_its_letter():
    tw = Typewriter("The monster escaped to the right side.", ELDER)
    hits = tw.phrase_hits(0)
    t0, y = synth_phrase(hits, ELDER, tw.text)
    assert t0 == hits[0].time
    assert np.isfinite(y).all() and 0.05 < np.max(np.abs(y)) < 1.0
    expected = hits[-1].time + hits[-1].duration - t0
    assert abs(len(y) / SR - expected) < 0.1


def test_syllables_stay_audible():
    """Continuous voice must still pulse per syllable (dips), not become one flat hum."""
    tw = Typewriter("la la la la la la", ELDER)
    t0, y = synth_phrase(tw.phrase_hits(0), ELDER, tw.text)
    hop = int(0.005 * SR)
    env = np.array([np.sqrt(np.mean(y[i:i + hop] ** 2)) for i in range(0, len(y) - hop, hop)])
    body = env[int(0.05 / 0.005):-int(0.1 / 0.005)]
    assert body.min() < 0.7 * body.max()


def test_render_deterministic():
    a = render_line("Sister Agnes, is that you?", ELDER)
    assert np.array_equal(a, render_line("Sister Agnes, is that you?", ELDER))
