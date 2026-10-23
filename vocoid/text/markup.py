"""Inline tags.

    "Meow{purr} feed me"        -> sound effect, fires on the letter before the tag
    "{mood:sad}I lost it all."  -> mood from here on; "{mood:happy:0.5}" = half strength
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

TAG = re.compile(r"\{([A-Za-z_][\w-]*)(?::([A-Za-z_][\w-]*))?(?::([\d.]+))?\}")


@dataclass
class Markup:
    text: str
    sfx: dict[int, list[str]] = field(default_factory=dict)
    moods: dict[int, tuple[str, float]] = field(default_factory=dict)  # char index -> (mood, strength)


def parse_tags(line: str) -> Markup:
    parts: list[str] = []
    length = 0
    out = Markup("")
    pos = 0
    for m in TAG.finditer(line):
        chunk = line[pos:m.start()]
        parts.append(chunk)
        length += len(chunk)
        name, arg, amount = m.groups()
        if name == "mood" and arg:
            out.moods[length] = (arg, float(amount) if amount else 1.0)  # applies from next char
        else:
            out.sfx.setdefault(max(0, length - 1), []).append(name)
        pos = m.end()
    parts.append(line[pos:])
    out.text = "".join(parts)
    return out


def parse_markup(line: str) -> tuple[str, dict[int, list[str]]]:
    """Text + sound effects only (mood tags are stripped)."""
    m = parse_tags(line)
    return m.text, m.sfx
