import pytest

from domain_shift_forgetting.data.classes import token_class
from domain_shift_forgetting.data.dedup import DocumentIdentity, candidate_jaccard, exact_dedup
from domain_shift_forgetting.data.packing import PackingAccounting, TokenDocument, pack_documents, rare_flags
from domain_shift_forgetting.data.splits import RepositoryDocument, repository_groups


def test_packing_scores_boundary_labels_once_and_attributes_eos():
    accounting = PackingAccounting()
    docs = [TokenDocument("a", "dev", (10, 11)), TokenDocument("b", "dev", (20, 21, 22))]
    windows = list(pack_documents(docs, split="dev", accounting=accounting, context=3))
    assert [w.labels for w in windows] == [(11, 50256, 20), (21, 22, 50256)]
    assert windows[0].label_document_ids == ("a", "a", "b")
    assert windows[1].inputs[0] == windows[0].labels[-1]
    assert accounting.source_tokens_including_eos == 7
    assert accounting.supervised_labels == 6
    assert accounting.terminal_unscored_tokens == 0
    assert accounting.complete


def test_mixed_split_rejected_and_short_tail_accounted():
    with pytest.raises(ValueError):
        list(pack_documents([TokenDocument("a", "test", (1,))], split="dev", accounting=PackingAccounting()))
    a = PackingAccounting()
    assert list(pack_documents([TokenDocument("a", "dev", (1, 2))], split="dev", accounting=a)) == []
    assert a.initial_context_tokens == 1 and a.terminal_unscored_tokens == 2


@pytest.mark.parametrize("token,raw,expected", [
    (50256, b"<|endoftext|>", "X"), (1, b" \n\t\r\f", "W"),
    (2, b" abc!", "A"), (3, b";{}", "P"), (4, b"\xff", "X"),
    (5, b"abc\x00", "X"), (6, b"\v", "X"),
])
def test_disjoint_byte_classes(token, raw, expected):
    assert token_class(token, raw) == expected


def test_repository_alias_transitivity_and_pool_order_invariance():
    rows = [RepositoryDocument("a", ("repo/a",)), RepositoryDocument("b", ("repo/b",)),
            RepositoryDocument("bridge", ("repo/a", "repo/b"))]
    groups = repository_groups(rows)
    assert len(set(groups.values())) == 1
    assert groups == repository_groups(reversed(rows))
    assert repository_groups(rows[:2])["a"] != groups["a"]


def test_exact_dedup_preserves_test_across_domains():
    rows = [DocumentIdentity("train", "a" * 64, "web", "train"),
            DocumentIdentity("dev", "a" * 64, "python", "dev"),
            DocumentIdentity("test", "a" * 64, "web", "test")]
    kept, removed = exact_dedup(rows)
    assert [r.document_id for r in kept] == ["test"]
    assert {r.removed_id for r in removed} == {"dev", "train"}
    assert candidate_jaccard([1, 2], [1, 2]) is None
    assert candidate_jaccard([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1


def test_rare_count_does_not_double_count_boundary_context():
    windows = list(pack_documents([TokenDocument("a", "train", (7,) * 99)], split="train",
                                 accounting=PackingAccounting(), context=1))
    assert 7 in rare_flags(windows)  # Exactly 99 unique positions, not 197.
    windows = list(pack_documents([TokenDocument("a", "train", (7,) * 100)], split="train",
                                 accounting=PackingAccounting(), context=1))
    assert 7 not in rare_flags(windows)
