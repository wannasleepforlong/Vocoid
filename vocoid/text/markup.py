"""Inline sound-effect tags: "Meow{purr} feed me" -> display text + {letter index: [sfx]}."""

import re

TAG = re.compile(r"\{([A-Za-z_][\w-]*)\}")


def parse_markup(line: str) -> tuple[str, dict[int, list[str]]]:
    """Tag fires when the letter right before it is typed (index 0 if at start)."""
    parts: list[str] = []
    length = 0
    sfx: dict[int, list[str]] = {}
    pos = 0
    for m in TAG.finditer(line):
        chunk = line[pos:m.start()]
        parts.append(chunk)
        length += len(chunk)
        sfx.setdefault(max(0, length - 1), []).append(m.group(1))
        pos = m.end()
    parts.append(line[pos:])
    return "".join(parts), sfx
