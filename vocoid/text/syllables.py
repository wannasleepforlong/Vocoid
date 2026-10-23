"""Syllables from real pronunciation (CMU dictionary), with letter-rule fallback for made-up names.

Each syllable knows its vowel (and the vowel it slides to, for "ai"-type diphthongs), its stress,
and the consonants around it. The rhythm and the sound are built from these, not from spelling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")

# ARPAbet vowel -> (vowel, vowel it slides to or None)
ARPA_VOWEL: dict[str, tuple[str, str | None]] = {
    "AA": ("ah", None),   # father, monster
    "AE": ("ae", None),   # cat, Agnes
    "AH": ("uh", None),   # but, the
    "AO": ("aw", None),   # thought
    "EH": ("eh", None),   # bed
    "ER": ("er", None),   # her, -er
    "IH": ("ih", None),   # sit
    "IY": ("ee", None),   # see
    "UH": ("uu", None),   # book
    "UW": ("ooh", None),  # to, boot
    "AY": ("ah", "ee"),   # right, side
    "EY": ("eh", "ee"),   # escaped, day
    "OW": ("oh", "ooh"),  # go
    "AW": ("ah", "ooh"),  # now
    "OY": ("aw", "ee"),   # boy
}
VOICELESS = {"P", "T", "K", "F", "S", "SH", "TH", "HH", "CH"}

FUNCTION_WORDS = {
    "a", "an", "the", "to", "of", "and", "or", "but", "in", "on", "at", "by", "for", "with", "from",
    "is", "am", "are", "was", "were", "be", "been", "it", "its", "as", "if", "so", "than", "that",
    "this", "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "his", "our", "their", "do", "does", "did", "has", "have", "had", "can", "will", "would", "not",
}

# letter fallback
LETTER_VOWEL = {"a": "ah", "e": "eh", "i": "ih", "o": "oh", "u": "uh", "y": "ee"}
DIGRAPH_VOWEL: dict[str, tuple[str, str | None]] = {
    "oo": ("ooh", None), "ou": ("ah", "ooh"), "ue": ("ooh", None), "ui": ("ooh", None),
    "ee": ("ee", None), "ea": ("ee", None), "ie": ("ee", None), "ei": ("eh", "ee"), "ey": ("eh", "ee"),
    "ai": ("eh", "ee"), "ay": ("eh", "ee"),
    "oa": ("oh", "ooh"), "oe": ("oh", None), "oi": ("aw", "ee"), "oy": ("aw", "ee"),
    "au": ("aw", None),
}
VOICELESS_LETTERS = set("ptkfsxch")


@dataclass(frozen=True)
class Syllable:
    index: int                 # char index in text where this syllable's sound triggers
    vowel: str
    word: int                  # word number in the line
    vowel_end: str | None = None  # diphthong target
    stress: int = 1            # 0 unstressed, 1 primary, 2 secondary
    onset: tuple[str, ...] = field(default=())  # consonants before the vowel
    coda: tuple[str, ...] = field(default=())   # consonants after it
    function_word: bool = False

    @property
    def voiceless_onset(self) -> int:
        return sum(c in VOICELESS for c in self.onset)

    @property
    def voiceless_coda(self) -> int:
        return sum(c in VOICELESS for c in self.coda)


@lru_cache(maxsize=1)
def _cmu() -> dict | None:
    try:
        import cmudict
        return cmudict.dict()
    except Exception:  # no dictionary installed: letter rules only
        return None


def _split_phones(phones: list[str]) -> list[tuple[tuple[str, ...], str, int, tuple[str, ...]]]:
    """ARPAbet phones -> [(onset, vowel phone, stress, coda)]. Between vowels: one consonant goes
    to the next syllable; with a cluster, the first stays behind (mon|ster)."""
    vowels = [i for i, p in enumerate(phones) if p[-1].isdigit()]
    out = []
    for k, vi in enumerate(vowels):
        prev_v = vowels[k - 1] if k else -1
        next_v = vowels[k + 1] if k + 1 < len(vowels) else len(phones)
        left = phones[prev_v + 1:vi]
        right = phones[vi + 1:next_v]
        onset = left if k == 0 else (left[1:] if len(left) >= 2 else left)
        coda = right if k + 1 == len(vowels) else (right[:1] if len(right) >= 2 else [])
        out.append((tuple(onset), phones[vi][:-1], int(phones[vi][-1]), tuple(coda)))
    return out


def _vowel_groups(text: str, start: int, end: int) -> list[tuple[int, str]]:
    groups, i = [], start
    while i < end:
        if _is_vowel(text, i, start):
            j = i
            while j < end and _is_vowel(text, j, start):
                j += 1
            groups.append((i, text[i:j].lower()))
            i = j
        else:
            i += 1
    return groups


def _is_vowel(text: str, i: int, word_start: int) -> bool:
    c = text[i].lower()
    return c in "aeiou" or (c == "y" and i > word_start)


def _char_positions(groups: list[tuple[int, str]], n: int, start: int, end: int) -> list[int]:
    """Pick n trigger chars in the word: vowel-group starts, trimmed or padded to n."""
    pos = [g[0] for g in groups][:n]
    if not pos:
        pos = [start]
    while len(pos) < n:  # more spoken syllables than spelled vowel groups
        last = pos[-1]
        pos.append(min(end - 1, last + max(1, (end - last) // 2)))
    return pos


def _from_dict(text, start, end, w, phones, func) -> list[Syllable]:
    parts = _split_phones(phones)
    pos = _char_positions(_vowel_groups(text, start, end), len(parts), start, end)
    out = []
    for p, (onset, v, stress, coda) in zip(pos, parts):
        vowel, vowel_end = ARPA_VOWEL.get(v, ("uh", None))
        out.append(Syllable(p, vowel, w, vowel_end, 0 if func else stress, onset, coda, func))
    return out


def _from_letters(text, start, end, w, func) -> list[Syllable]:
    groups = _vowel_groups(text, start, end)
    if len(groups) > 1 and groups[-1][1] == "e" and groups[-1][0] == end - 1:
        groups.pop()  # silent final e
    out = []
    for k, (gi, g) in enumerate(groups):
        nxt = groups[k + 1][0] if k + 1 < len(groups) else end
        prv_end = (groups[k - 1][0] + len(groups[k - 1][1])) if k else start
        onset = tuple(c.upper() for c in text[prv_end:gi].lower() if c.isalpha())
        onset = onset[1:] if k and len(onset) >= 2 else onset
        coda = tuple(c.upper() for c in text[gi + len(g):nxt].lower() if c.isalpha())
        coda = coda if k + 1 == len(groups) else (coda[:1] if len(coda) >= 2 else ())
        vowel, vowel_end = DIGRAPH_VOWEL.get(g[:2], (LETTER_VOWEL.get(g[0], "uh"), None))
        stress = 0 if func else (1 if k == 0 else 0)
        voiceless = tuple("S" if c.lower() in VOICELESS_LETTERS else c for c in onset)
        out.append(Syllable(gi, vowel, w, vowel_end, stress, voiceless, coda, func))
    return out


def find_syllables(text: str, use_dict: bool = True) -> list[Syllable]:
    cmu = _cmu() if use_dict else None
    out: list[Syllable] = []
    for w, m in enumerate(WORD.finditer(text)):
        start, end = m.span()
        word = m.group().lower()
        func = word in FUNCTION_WORDS
        phones = cmu.get(word) if cmu else None
        if phones:
            out.extend(_from_dict(text, start, end, w, phones[0], func))
        else:
            out.extend(_from_letters(text, start, end, w, func))
    return out


def words(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group()) for m in WORD.finditer(text)]
