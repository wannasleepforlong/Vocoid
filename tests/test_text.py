from vocoid.text.markup import parse_markup
from vocoid.text.prosody import plan_line, split_phrases
from vocoid.text.rhythm import timeline
from vocoid.text.syllables import find_syllables


def vowels(text, use_dict=True):
    return [s.vowel + (">" + s.vowel_end if s.vowel_end else "") for s in find_syllables(text, use_dict)]


def test_real_pronunciation():
    # the user's sentence: spelling-based rules got "to", "right", "side", "escaped" wrong
    assert vowels("The monster escaped to the right side") == [
        "uh", "ah", "er", "ih", "eh>ee", "ooh", "uh", "ah>ee", "ah>ee"]
    assert vowels("Sister Agnes") == ["ih", "er", "ae", "ih"]


def test_syllable_trigger_letters():
    sy = find_syllables("monster escaped")
    assert [s.index for s in sy] == [1, 5, 8, 11]  # mOnstEr | Escaped  (no syllable for "-ed")
    assert sy[0].stress == 1 and sy[1].stress == 0
    assert sy[1].onset == ("S", "T") and sy[0].coda == ("N",)


def test_function_words_marked():
    sy = find_syllables("to the castle")
    assert [s.function_word for s in sy] == [True, True, False, False]


def test_letter_fallback_for_unknown_words():
    assert vowels("Zorblax", use_dict=True) == vowels("Zorblax", use_dict=False) == ["oh", "ah"]
    assert vowels("yes", use_dict=False) == ["eh"]          # word-initial y is a consonant
    assert vowels("happy", use_dict=False) == ["ah", "ee"]  # trailing y is a vowel
    assert vowels("make", use_dict=False) == ["ah"]         # silent final e
    assert vowels("boot", use_dict=False) == ["ooh"]


def _timeline(text):
    sy = find_syllables(text)
    notes = plan_line(text, sy)
    return timeline(text, sy, notes, split_phrases(text))


def test_rhythm_follows_speech_not_spelling():
    # "monster" (7 letters, 2 syllables) and "a" must not be timed by letter count
    sounds, times = _timeline("The monster escaped to the right side")
    starts = [s.time for s in sorted(sounds.values(), key=lambda s: s.time)]
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert max(gaps) < 0.45 and min(gaps) > 0.06       # one even stream, no word holes
    assert 3.0 < len(starts) / (times[-1] - times[0] + 0.3) < 6.5  # syllables per second, like speech
    assert times == sorted(times)


def test_letters_hit_with_their_sound():
    sounds, times = _timeline("Hello there, friend.")
    for i, s in sounds.items():
        assert times[i] == s.time


def test_pause_only_at_punctuation():
    sounds, _ = _timeline("one two, three")
    a, b, c = sorted(sounds.values(), key=lambda s: s.time)
    assert (c.time - b.time) > (b.time - a.time) + 0.15  # comma pause


def test_markup():
    text, sfx = parse_markup("Mrrow.{purr} Hi{chime}")
    assert text == "Mrrow. Hi"
    assert sfx == {5: ["purr"], 8: ["chime"]}
    assert parse_markup("{purr}Hi")[1] == {0: ["purr"]}


def test_phrases():
    kinds = [p.kind for p in split_phrases("Oh, really? Wait... no! Fine")]
    assert kinds == [",", "?", "...", "!", ""]


def _ordered(text):
    return [n for _, n in sorted(plan_line(text).items())]


def test_drift_resets_at_comma():
    notes = _ordered("one two three four, five six seven eight")
    before = [n for n in notes if n.index < 18]
    after = [n for n in notes if n.index > 18]
    assert before[0].pitch > before[-1].pitch
    assert after[0].pitch > before[-1].pitch


def test_question_rises():
    q = _ordered("is that really you?")
    s = _ordered("is that really you.")
    assert q[-1].pitch > s[-1].pitch + 0.2


def test_shout_and_trail():
    caps = _ordered("I am REAL now")  # REAL = R IY1 L
    real = [n for n in caps if 5 <= n.index <= 8]
    assert real and real[0].gain > 1.3
    trail = _ordered("so very very tired...")
    assert trail[-1].gain < trail[0].gain * 0.5


def test_deterministic():
    assert plan_line("Same line, same tune?") == plan_line("Same line, same tune?")
