from __future__ import annotations

import re

_DEFAULT_WORDS = {
    "shit",
    "fuck",
    "fucking",
    "bitch",
    "asshole",
    "damn",
    "bastard",
}


def mask_profanity(text: str, *, mask_char: str = "*") -> str:
    if not text:
        return text

    pattern = re.compile(r"\b(" + "|".join(re.escape(w) for w in sorted(_DEFAULT_WORDS, key=len, reverse=True)) + r")\b", re.IGNORECASE)

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0)
        return mask_char * len(token)

    return pattern.sub(_replace, text)
