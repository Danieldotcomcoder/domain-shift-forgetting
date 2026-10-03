"""Streaming packing with document attribution and a one-token context overlap."""

from collections import Counter, deque
from dataclasses import dataclass
from typing import Iterable, Iterator


@dataclass(frozen=True)
class TokenDocument:
    document_id: str
    split: str
    tokens: tuple[int, ...]  # Ordinary-text encoding, without appended EOS.


@dataclass(frozen=True)
class PackedWindow:
    inputs: tuple[int, ...]
    labels: tuple[int, ...]
    label_document_ids: tuple[str, ...]
    source_start: int  # Position of the context token in this split's source stream.


@dataclass
class PackingAccounting:
    source_tokens_including_eos: int = 0
    supervised_labels: int = 0
    initial_context_tokens: int = 0
    terminal_unscored_tokens: int = 0
    complete: bool = False


def pack_documents(documents: Iterable[TokenDocument], *, split: str,
                   accounting: PackingAccounting, context: int = 512,
                   eos: int = 50256) -> Iterator[PackedWindow]:
    """Callers must exhaust the iterator before treating accounting as complete.

    Remainders cross document boundaries, never split boundaries. Each source
    position after the first is scored once unless in a terminal short window.
    """
    if context <= 0 or split not in {"train", "dev", "test"}:
        raise ValueError("Invalid context or split")
    if accounting != PackingAccounting():
        raise ValueError("Use fresh accounting for each packing stream")
    pending: deque[tuple[int, str]] = deque()
    seen: set[str] = set()
    for doc in documents:
        if doc.split != split or not doc.document_id or doc.document_id in seen:
            raise ValueError("Mixed split, empty document ID, or repeated document")
        seen.add(doc.document_id)
        for token in (*doc.tokens, eos):
            if not isinstance(token, int) or not 0 <= token < 50257:
                raise ValueError("Token outside GPT-2 vocabulary")
            pending.append((token, doc.document_id))
            accounting.source_tokens_including_eos += 1
            if len(pending) == context + 1:
                window = tuple(pending)
                accounting.supervised_labels += context
                yield PackedWindow(tuple(t for t, _ in window[:-1]),
                                   tuple(t for t, _ in window[1:]),
                                   tuple(d for _, d in window[1:]),
                                   accounting.supervised_labels - context)
                for _ in range(context):
                    pending.popleft()
    accounting.initial_context_tokens = int(accounting.source_tokens_including_eos > 0)
    accounting.terminal_unscored_tokens = max(0, len(pending) - 1)
    accounting.complete = True


def rare_flags(prefix_windows: Iterable[PackedWindow]) -> frozenset[int]:
    """Count unique actual-prefix source positions, including context positions.

    Supply exactly the windows selected for this seed's prefix, sorted by
    source_start from one packed source stream. Disconnected selected windows
    are allowed; adjoining windows share one context position counted once.
    """
    counts: Counter[int] = Counter()
    last_position = -1
    last_token = None
    for window in prefix_windows:
        if not window.inputs or len(window.inputs) != len(window.labels):
            raise ValueError("Malformed window")
        if window.source_start < 0 or window.source_start < last_position:
            raise ValueError("Require unique windows sorted by source position")
        if window.source_start > last_position:
            counts[window.inputs[0]] += 1
        elif window.inputs[0] != last_token:
            raise ValueError("Conflicting token at a shared source position")
        counts.update(window.labels)
        last_position = window.source_start + len(window.labels)
        last_token = window.labels[-1]
    return frozenset(token for token in range(50257) if counts[token] < 100)
