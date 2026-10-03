"""Classify pinned tokenizer bytes, not decoder replacement-character text."""

import unicodedata
from typing import Mapping


WHITESPACE_BYTES = frozenset(b" \t\n\r\f")


def token_class(token_id: int, token_bytes: bytes, *, eos: int = 50256) -> str:
    if token_id == eos or not token_bytes:
        return "X"
    try:
        text = token_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return "X"
    # The five whitespace bytes are permitted in W; other C-category code
    # points are treated as control/format/non-text (an explicit draft convention).
    if any(unicodedata.category(c).startswith("C") and ord(c) not in WHITESPACE_BYTES
           for c in text):
        return "X"
    if all(byte in WHITESPACE_BYTES for byte in token_bytes):
        return "W"
    if any(c.isalnum() for c in text):
        return "A"
    return "P"


def class_table(decoded_bytes: Mapping[int, bytes]) -> tuple[str, ...]:
    if set(decoded_bytes) != set(range(50257)):
        raise ValueError("Require all 50,257 IDs from the pinned GPT-2 byte mapping")
    return tuple(token_class(i, decoded_bytes[i]) for i in range(50257))
