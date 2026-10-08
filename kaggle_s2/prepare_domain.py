#!/usr/bin/env python3
"""Study 2 data preparation for the selected domain X (CPU only; see study2/PROTOCOL.md, Sec. 5).

  python prepare_domain.py probe-windows --output DIR      # candidate probe samples for the selection probe
  python prepare_domain.py prepare --inputs /kaggle/input --output DIR --cache DIR
  python prepare_domain.py verify --prepared DIR --pilot-online DIR
  python prepare_domain.py package --prepared DIR --output DIR   # private-dataset folder, never the reserved test

This mirrors the pilot's preparation (src/domain_shift_forgetting/pilot_data.py and pilot_arrays.py): the same
GPT-2 tokenizer (verified by its id-to-bytes hash), one EOS per document, 513-token windows at stride 512, the
same quotas, split conventions, exact + MinHash/LSH deduplication settings and order generator. One addition:
X is deduplicated against the pilot's frozen arrays, which can no longer change and therefore always win.
Raw text is streamed from Hugging Face at pinned revisions and never stored; only GPT-2 tokens are kept.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import asdict, dataclass
import gzip
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import random
import re
import shutil
import sqlite3
import struct
import sys
import time
import unicodedata

import numpy as np

SAMPLING_SEED = 20260911     # the pilot's corpus seed: shard order, document order, MinHash and audit sampling
X_ORDER_ENTROPY = 20261008   # X training order of seed s: np.random.default_rng([X_ORDER_ENTROPY, s])
EOS, CONTEXT, VOCAB = 50256, 512, 50257
SEQUENCES_PER_UPDATE, PREFIX_END, CONTINUATION_UPDATES = 32, 9156, 6104
PILOT_SEEDS, FRESH_SEEDS = (101, 102, 103), (104, 105, 106)
SEEDS = PILOT_SEEDS + FRESH_SEEDS
X_QUOTAS = {"x_train": 110_000_000, "x_dev": 2_097_153, "x_test": 8_388_609}
RAW_TARGET_MARGIN = 0.15     # stream until every split holds 1.15 x its quota in raw (pre-dedup) tokens
RAW_TARGET_STEP = 0.10       # if dedup leaves a split short, stream a further 10% of its quota, then refilter
MAX_FILTER_ROUNDS = 12
PROBE_WINDOWS = 256          # 131,072 positions: the size of the pilot's diagnostic probes
NEAR_DUPLICATE_JACCARD = 0.85
PILOT_ID_TO_BYTES_SHA256 = "a5623714bcf19049eb0fd78df19b6daa1e61ac435b86a5deae767768e7fcdc3d"
PILOT_ONLINE_MANIFEST_SHA256 = "290886d93760f5a66f1e55b3c4730a02dfddf1b8153ba12ed78fba6374ab87d7"
S2_ARRAYS_SCHEMA, S2_ORDERS_SCHEMA = "s2-arrays-v1", "s2-orders-v1"
SELECTION_SCHEMA = "s2-selection-v1"
HUB = "https://huggingface.co"
USER_AGENT = "domain-shift-forgetting-study2-preparation"

C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"    # allenai/c4: the pilot's own pinned revision
OWM_REVISION = "fde8ef8de2300f5e778f56261843dab89f230815"   # open-web-math/open-web-math


@dataclass(frozen=True)
class Candidate:
    id: str
    label: str
    repo: str
    revision: str
    format: str                              # "jsonl.gz" or "parquet"
    upstreams: tuple[tuple[str, str], ...]   # (upstream split, full-match regex over repository paths)
    split_rule: str                          # "c4-official" or "url-group"
    license: str


def _mc4(language: str, label: str, train_shards: int, validation_shards: int) -> Candidate:
    return Candidate(
        f"mc4-{language}", label, "allenai/c4", C4_REVISION, "jsonl.gz",
        (("train", rf"multilingual/c4-{language}\.tfrecord-\d{{5}}-of-{train_shards:05d}\.json\.gz"),
         ("validation", rf"multilingual/c4-{language}-validation\.tfrecord-\d{{5}}-of-{validation_shards:05d}\.json\.gz")),
        "c4-official", "ODC-BY 1.0 (allenai/c4 dataset card); Common Crawl terms of use")


CANDIDATES = {c.id: c for c in (
    _mc4("de", "German web text (mC4 de)", 2048, 16),
    _mc4("ru", "Russian web text (mC4 ru)", 4096, 32),
    _mc4("zh", "Chinese web text (mC4 zh)", 1024, 2),
    Candidate("openwebmath", "Mathematical web text (OpenWebMath)", "open-web-math/open-web-math", OWM_REVISION,
              "parquet", (("train", r"data/train-\d{5}-of-00114-[0-9a-f]+\.parquet"),), "url-group",
              "ODC-BY 1.0 (dataset card); Common Crawl terms of use"),
)}
CANDIDATE_ORDER = ("mc4-de", "mc4-ru", "mc4-zh", "openwebmath")  # also the tie-break order of the selection rule


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# =============================================================================
# Split rules and document identity (pilot conventions; see data/splits.py)
# =============================================================================
def c4_validation_split(content_sha256: str) -> str:
    """The pilot's rule for official C4 validation documents: SHA-256 of the content hash, modulo two."""
    return "dev" if int(sha256_text(content_sha256), 16) % 2 == 0 else "test"


def url_group(url: str | None, content_sha256: str) -> str:
    """Group identity for a source without an official held-out split: its URL (else its content)."""
    url = (url or "").strip()
    return sha256_text(f"url:{url}") if url else sha256_text(f"content:{content_sha256}")


def group_split(group_sha256: str) -> str:
    """The pilot's Python rule on a SHA-256 group identity: 0-89 train, 90-94 dev, 95-99 test."""
    bucket = int(group_sha256, 16) % 100
    return "train" if bucket < 90 else "dev" if bucket < 95 else "test"


def assign_split(candidate: Candidate, upstream: str, content_sha256: str, url: str | None) -> str:
    if candidate.split_rule == "c4-official":
        return "train" if upstream == "train" else c4_validation_split(content_sha256)
    if candidate.split_rule == "url-group":
        return group_split(url_group(url, content_sha256))
    raise ValueError(f"Unknown split rule {candidate.split_rule}")


def split_feeds(candidate: Candidate, upstream: str) -> set[str]:
    if candidate.split_rule == "c4-official":
        return {"train"} if upstream == "train" else {"dev", "test"}
    return {"train", "dev", "test"}


def document_id(revision: str, shard: str, row: int) -> str:
    return hashlib.sha256(f"{revision}:{shard}:{row}".encode()).hexdigest()


def eligible(text) -> bool:
    return isinstance(text, str) and bool(text.strip())


def shard_order(paths) -> list[str]:
    """The pilot's deterministic shard order."""
    return sorted(paths, key=lambda p: hashlib.sha256(f"{SAMPLING_SEED}:{p}".encode()).hexdigest())


# =============================================================================
# Hugging Face access (public datasets; no token; plain HTTPS at pinned revisions)
# =============================================================================
def _session():
    import requests
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def hub_files(repo: str, revision: str) -> list[str]:
    response = _session().get(f"{HUB}/api/datasets/{repo}/revision/{revision}", timeout=120)
    response.raise_for_status()
    info = response.json()
    if info.get("sha") != revision:
        raise ValueError(f"{repo}: the hub did not resolve revision {revision}")
    return [s["rfilename"] for s in info.get("siblings", [])]


def upstream_files(candidate: Candidate, listing: list[str] | None = None) -> dict[str, list[str]]:
    listing = hub_files(candidate.repo, candidate.revision) if listing is None else listing
    files = {name: shard_order(p for p in listing if re.fullmatch(pattern, p))
             for name, pattern in candidate.upstreams}
    for name, paths in files.items():
        if not paths:
            raise ValueError(f"{candidate.id}: no {name} shards at revision {candidate.revision}")
    return files


def paths_info(candidate: Candidate, paths: list[str]) -> dict[str, dict]:
    """Size and LFS SHA-256 of each file at the pinned revision (downloads are verified against them)."""
    result, session = {}, _session()
    for start in range(0, len(paths), 50):
        response = session.post(f"{HUB}/api/datasets/{candidate.repo}/paths-info/{candidate.revision}",
                                data={"paths": paths[start:start + 50]}, timeout=120)
        response.raise_for_status()
        for row in response.json():
            result[row["path"]] = {"size": row.get("size"), "sha256": (row.get("lfs") or {}).get("oid")}
    return result


def resolve_url(candidate: Candidate, path: str) -> str:
    return f"{HUB}/datasets/{candidate.repo}/resolve/{candidate.revision}/{path}"


def download(candidate: Candidate, path: str, cache: Path, expected: dict | None = None) -> Path:
    """Whole-file download into the cache, verified against the hub's size/SHA-256 when known."""
    target = cache / candidate.repo.replace("/", "__") / candidate.revision / path
    if target.exists() and (expected is None or expected.get("sha256") in (None, sha256_file(target))):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    session = _session()
    for attempt in range(5):
        try:
            with session.get(resolve_url(candidate, path), stream=True, timeout=120) as response:
                response.raise_for_status()
                digest, size = hashlib.sha256(), 0
                with partial.open("wb") as stream:
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        stream.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
            if expected and expected.get("size") is not None and size != expected["size"]:
                raise OSError(f"size {size} != {expected['size']}")
            if expected and expected.get("sha256") and digest.hexdigest() != expected["sha256"]:
                raise OSError("SHA-256 mismatch")
            partial.replace(target)
            return target
        except Exception as exc:  # retry transient network failures; never keep a partial file
            partial.unlink(missing_ok=True)
            log(f"download {path} failed ({type(exc).__name__}: {exc}); attempt {attempt + 1}/5")
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"Could not download {path}")


def iter_local_rows(candidate: Candidate, local: Path):
    """(row index, text, url, extra metadata) in file order; the row index is the pilot's shard row identity."""
    if candidate.format == "jsonl.gz":
        with gzip.open(local, "rt", encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                row = json.loads(line)
                yield index, row.get("text"), row.get("url"), {"timestamp": row.get("timestamp")}
    else:
        import pyarrow.parquet as pq
        index = 0
        for batch in pq.ParquetFile(local).iter_batches(batch_size=1024, columns=["text", "url", "date"]):
            columns = batch.to_pydict()
            for text, url, date in zip(columns["text"], columns["url"], columns["date"]):
                yield index, text, url, {"date": date}
                index += 1


def iter_streamed_rows(candidate: Candidate, path: str, cache: Path):
    """Rows of one shard; JSONL.gz shards are streamed (a prefix suffices for the probe), Parquet is downloaded."""
    if candidate.format != "jsonl.gz":
        yield from iter_local_rows(candidate, download(candidate, path, cache))
        return
    with _session().get(resolve_url(candidate, path), stream=True, timeout=120) as response:
        response.raise_for_status()
        response.raw.decode_content = False  # the file itself is gzip; decompress exactly once below
        with gzip.GzipFile(fileobj=response.raw) as raw, io.TextIOWrapper(raw, encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                row = json.loads(line)
                yield index, row.get("text"), row.get("url"), {"timestamp": row.get("timestamp")}


# =============================================================================
# Tokenizer (identical to the pilot's; verified by the id-to-bytes hash)
# =============================================================================
def gpt2_encoder():
    import tiktoken
    return tiktoken.get_encoding("gpt2")


def tokenizer_identity(encoder) -> dict:
    byte_map = hashlib.sha256(b"".join(struct.pack("<I", len(encoder.decode_single_token_bytes(i)))
                                       + encoder.decode_single_token_bytes(i) for i in range(VOCAB))).hexdigest()
    return {"encoding": "gpt2", "id_to_bytes_sha256": byte_map, "tiktoken": importlib.metadata.version("tiktoken"),
            "unicode": unicodedata.unidata_version}


def require_pilot_tokenizer(encoder) -> dict:
    identity = tokenizer_identity(encoder)
    if identity["id_to_bytes_sha256"] != PILOT_ID_TO_BYTES_SHA256:
        raise ValueError("The GPT-2 byte mapping differs from the pilot's; refusing to tokenize")
    return identity


# =============================================================================
# Selection-probe samples
# =============================================================================
def probe_sample(candidate: Candidate, encoder, cache: Path, listing: list[str] | None = None,
                 windows: int = PROBE_WINDOWS) -> tuple[np.ndarray, dict]:
    """The candidate's pre-registered probe sample: eligible train-side documents from its shards in the pilot's
    shard order, rows in file order, GPT-2 tokens + EOS, until windows x 512 + 1 tokens (cut mid-document)."""
    need = windows * CONTEXT + 1
    files = upstream_files(candidate, listing)["train"]
    pieces, documents, total = [], [], 0
    for shard in files:
        for index, text, url, _ in iter_streamed_rows(candidate, shard, cache):
            if not eligible(text):
                continue
            content = sha256_text(text)
            if assign_split(candidate, "train", content, url) != "train":
                continue  # the probe never reads a document the split rule holds out
            ids = encoder.encode_ordinary(text) + [EOS]
            take = min(len(ids), need - total)
            pieces.append(np.asarray(ids[:take], dtype="<u2"))
            documents.append({"id": document_id(candidate.revision, shard, index), "shard": shard, "row": index,
                              "content_sha256": content, "tokens": take, "truncated": take < len(ids)})
            total += take
            if total == need:
                tokens = np.concatenate(pieces)
                return tokens, {"candidate": candidate.id, "repo": candidate.repo, "revision": candidate.revision,
                                "windows": windows, "tokens": int(len(tokens)),
                                "tokens_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
                                "documents": documents}
    raise ValueError(f"{candidate.id}: not enough probe text")


def build_probe_windows(output: Path, cache: Path, candidates=CANDIDATE_ORDER) -> dict:
    encoder = gpt2_encoder()
    identity = require_pilot_tokenizer(encoder)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "s2-probe-windows-v1", "tokenizer_hashes": identity, "sampling_seed": SAMPLING_SEED,
                "rule": "eligible train-side documents, shards in sha256(f'{seed}:{path}') order, rows in file order; "
                        "GPT-2 tokens + EOS; first 256 x 512 + 1 tokens", "candidates": {}}
    listings = {}
    for cid in candidates:
        candidate = CANDIDATES[cid]
        if candidate.repo not in listings:
            listings[candidate.repo] = hub_files(candidate.repo, candidate.revision)
        tick = time.time()
        tokens, info = probe_sample(candidate, encoder, cache, listings[candidate.repo])
        path = output / f"{cid}.bin"
        tokens.astype("<u2").tofile(path)
        info |= {"file": path.name, "file_sha256": sha256_file(path)}
        manifest["candidates"][cid] = info
        log(f"probe sample {cid}: {len(info['documents'])} documents from {info['documents'][0]['shard']} "
            f"in {time.time() - tick:.0f}s")
    json_write(output / "manifest.json", manifest)
    return manifest


def load_probe_windows(root: Path) -> dict[str, np.ndarray]:
    manifest = read_json(root / "manifest.json")
    result = {}
    for cid, info in manifest["candidates"].items():
        if sha256_file(root / info["file"]) != info["file_sha256"]:
            raise ValueError(f"Probe sample hash mismatch: {cid}")
        result[cid] = np.fromfile(root / info["file"], dtype="<u2")
    return result


# =============================================================================
# The pilot's frozen corpus: exact hashes of every online document; token shingles of its dev documents.
# =============================================================================
def _shingles(blob: bytes) -> set[bytes]:
    tokens = np.frombuffer(blob, dtype="<u2")
    return {tokens[i:i + 5].tobytes() for i in range(max(0, len(tokens) - 4))}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b)


def _minhash(shingles: set[bytes]):
    from datasketch import MinHash
    signature = MinHash(num_perm=128, seed=SAMPLING_SEED)
    signature.update_batch(sorted(shingles))
    return signature


class FrozenReference:
    """The pilot's online corpus can no longer change, so it wins every duplicate decision against X."""

    def __init__(self, pilot_online: Path):
        from datasketch import MinHashLSH
        manifest = read_json(pilot_online / "manifest.json")
        self.exact: set[str] = set()
        for key in ("web_train", "python_train", "web_dev", "python_dev"):
            for doc in read_json(pilot_online / manifest["splits"][key]["documents"])["documents"]:
                self.exact.add(doc["content_sha256"])
        self.index = MinHashLSH(num_perm=128, params=(32, 4))
        self.shingles: dict[str, set[bytes]] = {}
        for key in ("web_dev", "python_dev"):
            row = manifest["splits"][key]
            tokens = np.fromfile(pilot_online / row["file"], dtype="<u2")
            for doc in read_json(pilot_online / row["documents"])["documents"]:
                sequence = tokens[doc["token_start"]:doc["token_start"] + doc["tokens"]]
                if not doc["truncated"] and len(sequence) and sequence[-1] == EOS:
                    sequence = sequence[:-1]  # the pilot shingled documents without their EOS
                shingles = _shingles(sequence.astype("<u2").tobytes())
                if shingles:
                    name = f"{key}:{doc['id']}"
                    self.shingles[name] = shingles
                    self.index.insert(name, _minhash(shingles))

    def near_duplicate(self, shingles: set[bytes], signature) -> str | None:
        for name in sorted(self.index.query(signature)):
            if _jaccard(shingles, self.shingles[name]) >= NEAR_DUPLICATE_JACCARD:
                return name
        return None


# =============================================================================
# Candidate pool -> deduplicated retained documents -> packed arrays
# =============================================================================
class RowReader:
    """Eligible rows of one upstream in the deterministic order: shards in the pilot's order, rows in file order."""

    def __init__(self, candidate: Candidate, upstream: str, shards: list[str], cache: Path, info: dict):
        self.candidate, self.upstream, self.shards, self.cache, self.info = candidate, upstream, shards, cache, info
        self.buffer: deque = deque()
        self.position = 0
        self.rows = None
        self.consumed_shards: list[str] = []
        self.dropped = Counter()

    def _next_raw(self):
        while True:
            if self.rows is None:
                if self.position == len(self.shards):
                    return None
                shard = self.shards[self.position]
                local = download(self.candidate, shard, self.cache, self.info.get(shard))
                self.consumed_shards.append(shard)
                self.rows = ((shard, *row) for row in iter_local_rows(self.candidate, local))
            row = next(self.rows, None)
            if row is not None:
                return row
            self.rows = None
            self.position += 1

    def take(self, count: int) -> list:
        batch = []
        while len(batch) < count:
            if self.buffer:
                batch.append(self.buffer.popleft())
                continue
            row = self._next_raw()
            if row is None:
                break
            if not eligible(row[2]):
                self.dropped["empty_text"] += 1
                continue
            batch.append(row)
        return batch

    def give_back(self, rows: list) -> None:
        self.buffer.extendleft(reversed(rows))


def open_pool(path: Path) -> sqlite3.Connection:
    path.unlink(missing_ok=True)
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE docs(id TEXT PRIMARY KEY, seq INTEGER, upstream TEXT, split TEXT, content_hash TEXT, "
               "tokens BLOB, metadata TEXT)")
    return db


def fill_pool(db, candidate: Candidate, readers: dict[str, RowReader], encoder, targets: dict[str, float],
              raw: Counter, sequence: list[int], batch_size: int = 2000) -> None:
    """Consume each upstream until every split it feeds holds its raw target; the stop is per document."""
    threads = max(1, os.cpu_count() or 1)
    for upstream, reader in readers.items():
        feeds = split_feeds(candidate, upstream)
        while any(raw[s] < targets[s] for s in feeds):
            batch = reader.take(batch_size)
            if not batch:
                raise ValueError(f"{candidate.id}/{upstream}: all shards consumed before the quotas were met")
            encoded = encoder.encode_ordinary_batch([row[2] for row in batch], num_threads=threads)
            for position, ((shard, index, text, url, extra), ids) in enumerate(zip(batch, encoded)):
                if not any(raw[s] < targets[s] for s in feeds):
                    reader.give_back(batch[position:])
                    break
                content = sha256_text(text)
                split = assign_split(candidate, upstream, content, url)
                doc_id = document_id(candidate.revision, shard, index)
                meta = {"dataset": candidate.repo, "revision": candidate.revision, "shard": shard, "row": index,
                        "content_sha256": content, "url": url, "split": split, "upstream": upstream,
                        "group_sha256": url_group(url, content) if candidate.split_rule == "url-group" else None,
                        **extra}
                db.execute("INSERT INTO docs VALUES (?,?,?,?,?,?,?)",
                           (doc_id, sequence[0], upstream, split, content, np.asarray(ids, dtype="<u2").tobytes(),
                            json.dumps(meta)))
                sequence[0] += 1
                raw[split] += len(ids) + 1
            db.commit()
        log(f"{upstream}: raw tokens {dict(raw)}; shards {reader.consumed_shards}")


def filter_pool(db, frozen: FrozenReference | None, audit_path: Path) -> tuple[dict[str, list[str]], dict]:
    """Exact and near-duplicate removal: frozen pilot documents win, then test over dev over train (ties by id)."""
    from datasketch import MinHashLSH
    rows = db.execute("SELECT id, split, content_hash FROM docs").fetchall()
    priority = {"test": 0, "dev": 1, "train": 2}
    rows.sort(key=lambda r: (priority[r[1]], r[0]))
    index = MinHashLSH(num_perm=128, params=(32, 4))
    seen, retained = {}, {f"x_{split}": [] for split in ("train", "dev", "test")}
    counts = Counter()
    with audit_path.open("w", encoding="utf-8") as removals:
        for doc_id, split, digest in rows:
            if frozen is not None and digest in frozen.exact:
                counts["exact_frozen"] += 1
                removals.write(json.dumps({"removed": doc_id, "retained": "pilot", "reason": "exact_frozen"}) + "\n")
                continue
            if digest in seen:
                counts["exact"] += 1
                removals.write(json.dumps({"removed": doc_id, "retained": seen[digest], "reason": "exact"}) + "\n")
                continue
            shingles = _shingles(db.execute("SELECT tokens FROM docs WHERE id=?", (doc_id,)).fetchone()[0])
            duplicate, reason, signature = None, None, None
            if shingles:
                signature = _minhash(shingles)
                if frozen is not None:
                    duplicate = frozen.near_duplicate(shingles, signature)
                    reason = "near_frozen_dev_jaccard_ge_0.85" if duplicate else None
                if duplicate is None:
                    for candidate in sorted(index.query(signature)):
                        other = _shingles(db.execute("SELECT tokens FROM docs WHERE id=?", (candidate,)).fetchone()[0])
                        if _jaccard(shingles, other) >= NEAR_DUPLICATE_JACCARD:
                            duplicate, reason = candidate, "near_jaccard_ge_0.85"
                            break
            else:
                counts["short_shingle_documents"] += 1
            if duplicate:
                counts[reason] += 1
                removals.write(json.dumps({"removed": doc_id, "retained": duplicate, "reason": reason}) + "\n")
                continue
            seen[digest] = doc_id
            retained[f"x_{split}"].append(doc_id)
            if signature is not None:
                index.insert(doc_id, signature)
    # A fixed random audit of retained cross-split pairs (and pairs with the frozen dev sets), outcome-independent.
    flat = [(key, doc) for key, docs in retained.items() for doc in docs]
    rng = random.Random(SAMPLING_SEED)
    checked = misses = frozen_checked = frozen_misses = 0
    for _ in range(min(10_000, len(flat) * 4)):
        a, b = rng.sample(flat, 2) if len(flat) >= 2 else (None, None)
        if a is None or a[0] == b[0]:
            continue
        sa, sb = (_shingles(db.execute("SELECT tokens FROM docs WHERE id=?", (x[1],)).fetchone()[0]) for x in (a, b))
        if sa and sb:
            checked += 1
            misses += int(_jaccard(sa, sb) >= NEAR_DUPLICATE_JACCARD)
    if frozen is not None and flat and frozen.shingles:
        names = sorted(frozen.shingles)
        for _ in range(2_000):
            _, doc = rng.choice(flat)
            mine = _shingles(db.execute("SELECT tokens FROM docs WHERE id=?", (doc,)).fetchone()[0])
            if mine:
                frozen_checked += 1
                frozen_misses += int(_jaccard(mine, frozen.shingles[rng.choice(names)]) >= NEAR_DUPLICATE_JACCARD)
    summary = {"exact_frozen_removed": counts["exact_frozen"], "exact_removed": counts["exact"],
               "near_frozen_dev_removed": counts["near_frozen_dev_jaccard_ge_0.85"],
               "near_removed": counts["near_jaccard_ge_0.85"],
               "short_shingle_documents": counts["short_shingle_documents"],
               "missed_candidate_pairs_checked": checked, "missed_candidate_pairs_found": misses,
               "frozen_pairs_checked": frozen_checked, "frozen_pairs_found": frozen_misses,
               "retained_documents": {k: len(v) for k, v in retained.items()},
               "minhash_permutations": 128, "seed": SAMPLING_SEED, "lsh_bands": 32, "lsh_rows": 4,
               "jaccard_threshold": NEAR_DUPLICATE_JACCARD}
    if misses or frozen_misses:
        raise ValueError("Missed-candidate audit found leakage; investigate before freezing")
    return retained, summary


def retained_tokens(db, ids: list[str]) -> int:
    return sum(db.execute("SELECT length(tokens)/2+1 FROM docs WHERE id=?", (doc,)).fetchone()[0] for doc in ids)


def quota_tokens(key: str, quotas=X_QUOTAS) -> int:
    """Tokens materialized per split: training keeps whole 512-label windows + the initial context."""
    wanted = quotas[key]
    return ((wanted - 1 + 511) // 512) * 512 + 1 if key.endswith("train") else wanted


def materialize(db, retained: dict, output: Path, quotas=X_QUOTAS) -> dict:
    """The pilot's packing: documents in sha256(f'{seed}:{id}') order, EOS appended, owners aligned 1:1 with tokens,
    whole training windows, exact development/test counts. The reserved test goes to a separate folder."""
    online, sealed = output / "s2online", output / "reserved-test"
    online.mkdir(parents=True, exist_ok=True)
    sealed.mkdir(parents=True, exist_ok=True)
    manifests = {online: {"schema": S2_ARRAYS_SCHEMA, "splits": {}},
                 sealed: {"schema": "s2-reserved-test-v1", "splits": {}}}
    for key, ids in retained.items():
        split = key.split("_")[1]
        target = sealed if split == "test" else online
        ids = sorted(ids, key=lambda value: hashlib.sha256(f"{SAMPLING_SEED}:{value}".encode()).hexdigest())
        token_path, owner_path = target / f"{key}.bin", target / f"{key}.owners.bin"
        wanted = quota_tokens(key, quotas)
        count, docs = 0, []
        with token_path.open("wb") as tokens, owner_path.open("wb") as owners:
            for doc_id in ids:
                blob, meta = db.execute("SELECT tokens,metadata FROM docs WHERE id=?", (doc_id,)).fetchone()
                values = np.concatenate((np.frombuffer(blob, dtype="<u2"), np.array([EOS], dtype="<u2")))
                take = min(len(values), wanted - count)
                values[:take].astype("<u2").tofile(tokens)
                np.full(take, len(docs), dtype="<u4").tofile(owners)
                docs.append({"id": doc_id, "token_start": count, "tokens": take,
                             "truncated": take < len(values), **json.loads(meta)})
                count += take
                if count == wanted:
                    break
        if count != wanted:
            raise ValueError(f"Post-filter quota incomplete: {key}: {count}/{wanted}")
        docs_path = target / f"{key}.documents.json"
        json_write(docs_path, {"documents": docs})
        manifests[target]["splits"][key] = {"tokens": count, "labels": count - 1, "file": token_path.name,
                                            "sha256": sha256_file(token_path), "owners": owner_path.name,
                                            "owners_sha256": sha256_file(owner_path), "documents": docs_path.name,
                                            "documents_sha256": sha256_file(docs_path), "initial_context_tokens": 1}
    json_write(sealed / "manifest.json", manifests[sealed])
    return manifests[online]


# =============================================================================
# Orders: the pilot's generator for web/Python (seeds 104-106), an independent stream for X (seeds 101-106).
# =============================================================================
def rare_flags(web_tokens: np.ndarray, prefix_windows: np.ndarray) -> np.ndarray:
    """The pilot's R flag: fewer than 100 occurrences in the seed's prefix positions (shared context once)."""
    counts = np.zeros(VOCAB, dtype=np.int64)
    selected = np.sort(prefix_windows)
    previous = -2
    for start in range(0, len(selected), 1024):
        indices = selected[start:start + 1024]
        offsets = indices[:, None] * CONTEXT + np.arange(CONTEXT + 1)
        rows = np.asarray(web_tokens[offsets], dtype=np.int64)
        counts += np.bincount(rows[:, 1:].ravel(), minlength=VOCAB)
        include = indices != np.concatenate(([previous], indices[:-1])) + 1
        counts += np.bincount(rows[include, 0], minlength=VOCAB)
        previous = indices[-1]
    return counts < 100


def pilot_orders(seed: int, web_tokens: np.ndarray, web_windows: int, python_windows: int) -> dict[str, np.ndarray]:
    """Exactly pilot_arrays.PilotArrays.make_orders for one seed."""
    rng = np.random.default_rng(seed)
    web = rng.permutation(web_windows)
    code = rng.permutation(python_windows)
    needed_web = (PREFIX_END + CONTINUATION_UPDATES) * SEQUENCES_PER_UPDATE
    needed_code = CONTINUATION_UPDATES * SEQUENCES_PER_UPDATE
    if len(web) < needed_web or len(code) < needed_code:
        raise ValueError("Insufficient unique windows; replacement sampling is forbidden")
    return {"web": web[:needed_web].astype("<u8"), "python": code[:needed_code].astype("<u8"),
            "rare": rare_flags(web_tokens, web[:PREFIX_END * SEQUENCES_PER_UPDATE])}


def x_order(seed: int, x_windows: int) -> np.ndarray:
    needed = CONTINUATION_UPDATES * SEQUENCES_PER_UPDATE
    if x_windows < needed:
        raise ValueError("Insufficient unique X windows; replacement sampling is forbidden")
    return np.random.default_rng([X_ORDER_ENTROPY, seed]).permutation(x_windows)[:needed].astype("<u8")


def _saved_sha256(values: np.ndarray, scratch: Path) -> str:
    np.save(scratch, values, allow_pickle=False)
    return sha256_file(scratch)


def make_orders(pilot_online: Path, pilot_orders_root: Path, s2_online: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    pilot_manifest = read_json(pilot_online / "manifest.json")
    s2_manifest = read_json(s2_online / "manifest.json")
    web_tokens = np.memmap(pilot_online / pilot_manifest["splits"]["web_train"]["file"], dtype="<u2", mode="r")
    web_windows = (len(web_tokens) - 1) // CONTEXT
    python_windows = (pilot_manifest["splits"]["python_train"]["tokens"] - 1) // CONTEXT
    x_windows = (s2_manifest["splits"]["x_train"]["tokens"] - 1) // CONTEXT
    pilot_order_manifest = read_json(pilot_orders_root / "manifest.json")
    manifest = {"schema": S2_ORDERS_SCHEMA,
                "pilot_array_manifest_sha256": sha256_file(pilot_online / "manifest.json"),
                "x_array_manifest_sha256": sha256_file(s2_online / "manifest.json"),
                "pilot_orders_manifest_sha256": sha256_file(pilot_orders_root / "manifest.json"),
                "rules": {"web_python_rare": "seeds 104-106: numpy default_rng(seed): permutation of web windows, "
                                            "then of Python windows (pilot_arrays.make_orders, unchanged)",
                          "x": f"seeds 101-106: numpy default_rng([{X_ORDER_ENTROPY}, seed]).permutation(x windows)"
                               f"[:{CONTINUATION_UPDATES * SEQUENCES_PER_UPDATE}]",
                          "seeds_101_103_web_python_rare": "the pilot's own order files (pilot dataset)"},
                "probe_rule": "last 256 training windows, forward-only", "numpy": np.__version__,
                "windows": {"web": web_windows, "python": python_windows, "x": x_windows},
                "seeds": {}, "reproduction_of_pilot_orders": {}}
    scratch = output / "scratch.npy"
    for seed in SEEDS:
        paths = {}
        orders = pilot_orders(seed, web_tokens, web_windows, python_windows)
        if seed in PILOT_SEEDS:
            # The same generator must reproduce the pilot's own files bit for bit; otherwise stop.
            reproduced = {key: _saved_sha256(values, scratch) == pilot_order_manifest["seeds"][str(seed)][key]["sha256"]
                          for key, values in orders.items()}
            manifest["reproduction_of_pilot_orders"][str(seed)] = reproduced
            if not all(reproduced.values()):
                raise ValueError(f"The order generator does not reproduce the pilot's seed {seed} orders: {reproduced}")
        else:
            for key, values in orders.items():
                path = output / f"{seed}-{key}.npy"
                np.save(path, values, allow_pickle=False)
                paths[key] = {"file": path.name, "sha256": sha256_file(path)}
        path = output / f"{seed}-x.npy"
        np.save(path, x_order(seed, x_windows), allow_pickle=False)
        paths["x"] = {"file": path.name, "sha256": sha256_file(path)}
        manifest["seeds"][str(seed)] = paths
        log(f"orders for seed {seed}: {sorted(paths)}")
    scratch.unlink(missing_ok=True)
    json_write(output / "manifest.json", manifest)
    return manifest


# =============================================================================
# Preparation driver
# =============================================================================
def locate(inputs: Path) -> dict[str, Path]:
    found = {"selection": [], "pilot_online": []}
    for path in inputs.rglob("*.json"):
        if path.name not in ("selection.json", "manifest.json"):
            continue
        try:
            schema = read_json(path).get("schema")
        except Exception:
            continue
        if path.name == "selection.json" and schema == SELECTION_SCHEMA:
            found["selection"].append(path)
        elif path.name == "manifest.json" and schema == "pilot-arrays-v1" and path.parent.name == "online":
            found["pilot_online"].append(path.parent)
    result = {}
    for key, paths in found.items():
        unique = sorted({p.resolve() for p in paths})
        if len(unique) != 1:
            raise FileNotFoundError(f"Expected exactly one {key} under {inputs}, found {unique}")
        result[key] = unique[0]
    result["pilot_orders"] = result["pilot_online"].parent / "orders"
    return result


def capture_upstream(candidate: Candidate, output: Path) -> dict:
    """Dataset card at the pinned revision (license/provenance evidence); no corpus text."""
    output.mkdir(parents=True, exist_ok=True)
    response = _session().get(resolve_url(candidate, "README.md"), timeout=120)
    response.raise_for_status()
    card = output / f"{candidate.id}-card.md"
    card.write_bytes(response.content)
    evidence = {"schema": 1, "status": "captured", "candidate": asdict(candidate),
                "files": {card.name: {"url": resolve_url(candidate, "README.md"), "sha256": sha256_file(card)}}}
    json_write(output / "source-evidence.json", evidence)
    return evidence


def load_selection(selection_path: Path, rank: int = 0) -> tuple[dict, Candidate]:
    """The selected candidate (rank 0). A higher rank is the protocol's only fallback (Sec. 5.9): the next candidate in
    the probe's ranking, used only if the selected one cannot meet its quotas, recorded as a deviation."""
    selection = read_json(selection_path)
    if selection.get("schema") != SELECTION_SCHEMA or selection.get("status") != "selected":
        raise ValueError("The selection probe did not produce a valid selection")
    if selection["ranking"][0] != selection["selected"]:
        raise ValueError("The selection's ranking does not start with its selected candidate")
    candidate = CANDIDATES[selection["ranking"][rank]]
    if selection["candidates"][candidate.id]["revision"] != candidate.revision:
        raise ValueError("The selection was made at a different source revision")
    return selection, candidate


def prepare(selection_path: Path, pilot_online: Path, pilot_orders_root: Path, output: Path, cache: Path,
            rank: int = 0) -> dict:
    """Bounded deterministic preparation of the selected domain; reports a deficit instead of repeating data."""
    selection, candidate = load_selection(selection_path, rank)
    if sha256_file(pilot_online / "manifest.json") != PILOT_ONLINE_MANIFEST_SHA256:
        raise ValueError("The attached pilot corpus is not the one the pilot ran on")
    if (output / "s2online" / "manifest.json").exists():
        raise FileExistsError("Preparation already completed; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    tick = time.time()
    encoder = gpt2_encoder()
    tokenizer = require_pilot_tokenizer(encoder)
    log(f"Preparing {candidate.id} ({candidate.repo} @ {candidate.revision[:12]}); tokenizer verified")
    frozen = FrozenReference(pilot_online)
    log(f"Frozen pilot reference: {len(frozen.exact):,} exact hashes, {len(frozen.shingles):,} dev documents "
        f"indexed ({time.time() - tick:.0f}s)")
    files = upstream_files(candidate)
    info = paths_info(candidate, [p for paths in files.values() for p in paths[:64]])
    readers = {upstream: RowReader(candidate, upstream, paths, cache, info) for upstream, paths in files.items()}
    db = open_pool(cache / "candidate-pool.sqlite")
    targets = {key.split("_")[1]: quota_tokens(key) * (1 + RAW_TARGET_MARGIN) for key in X_QUOTAS}
    raw, sequence, rounds = Counter(), [0], 0
    while True:
        fill_pool(db, candidate, readers, encoder, targets, raw, sequence)
        retained, audit = filter_pool(db, frozen, cache / "dedup-removals.jsonl")
        available = {key: retained_tokens(db, ids) for key, ids in retained.items()}
        short = [key for key in X_QUOTAS if available[key] < quota_tokens(key)]
        log(f"filter round {rounds}: retained tokens {available}; short: {short or 'none'}")
        if not short:
            break
        rounds += 1
        if rounds > MAX_FILTER_ROUNDS:
            raise ValueError(f"Quotas not met after {MAX_FILTER_ROUNDS} expansions: {available}")
        for key in short:
            targets[key.split("_")[1]] += X_QUOTAS[key] * RAW_TARGET_STEP
    shutil.copyfile(cache / "dedup-removals.jsonl", output / "dedup-removals.jsonl")
    capture_upstream(candidate, output / "s2upstream")
    provenance = {"shards_consumed": {u: r.consumed_shards for u, r in readers.items()},
                  "shard_sha256": {s: info.get(s, {}).get("sha256") for r in readers.values()
                                   for s in r.consumed_shards},
                  "dropped_rows": {u: dict(r.dropped) for u, r in readers.items()},
                  "raw_tokens_before_dedup": dict(raw), "filter_rounds": rounds + 1}
    manifest = finalize(db, retained, audit, selection_path=selection_path, pilot_online=pilot_online,
                        pilot_orders_root=pilot_orders_root, output=output, tokenizer=tokenizer, provenance=provenance,
                        rank=rank)
    db.close()
    log(f"Prepared {candidate.id} in {(time.time() - tick) / 60:.1f} min")
    return manifest


def finalize(db, retained: dict, audit: dict, *, selection_path: Path, pilot_online: Path, pilot_orders_root: Path,
             output: Path, tokenizer: dict, provenance: dict, quotas=X_QUOTAS, rank: int = 0) -> dict:
    """Pack the retained documents, write the online manifest the runner verifies, and generate the orders.
    Expects ``output/s2upstream/source-evidence.json`` (capture_upstream)."""
    selection, candidate = load_selection(selection_path, rank)
    pilot_manifest = read_json(pilot_online / "manifest.json")
    manifest = materialize(db, retained, output, quotas)
    online = output / "s2online"
    json_write(online / "audit.json", audit)
    shutil.copyfile(selection_path, online / "selection.json")
    sealed = read_json(output / "reserved-test" / "manifest.json")
    manifest |= {
        "domain": {"id": candidate.id, "label": candidate.label, "repo": candidate.repo,
                   "revision": candidate.revision, "format": candidate.format, "split_rule": candidate.split_rule,
                   "license": candidate.license, **{k: provenance[k] for k in
                                                    ("shards_consumed", "shard_sha256", "dropped_rows")}},
        "selection": {"selection_sha256": sha256_file(online / "selection.json"), "rank": rank,
                      "statistic": selection["candidates"][candidate.id]["S"],
                      "python_reference": selection["python_reference"]["S"]},
        "pilot_array_manifest_sha256": sha256_file(pilot_online / "manifest.json"),
        "tokenizer_hashes": tokenizer, "classes_sha256": pilot_manifest["classes_sha256"],
        "sampling_seed": SAMPLING_SEED, "quotas": dict(quotas),
        "raw_tokens_before_dedup": provenance["raw_tokens_before_dedup"], "filter_rounds": provenance["filter_rounds"],
        "dedup_audit_sha256": sha256_file(online / "audit.json"),
        "reserved_test_sealed": {"manifest_sha256": sha256_file(output / "reserved-test" / "manifest.json"),
                                 "labels": sealed["splits"]["x_test"]["labels"], "online_path_included": False},
        "upstream_evidence_sha256": sha256_file(output / "s2upstream" / "source-evidence.json"),
        "preparation_code_sha256": sha256_file(Path(__file__)),
        "preparation_packages": {name: _version(name) for name in
                                 ("numpy", "tiktoken", "pyarrow", "datasketch", "requests")},
        "preparation_status": "prepared_requires_acceptance_validation"}
    json_write(online / "manifest.json", manifest)
    make_orders(pilot_online, pilot_orders_root, online, output / "s2orders")
    return manifest


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


# =============================================================================
# Acceptance validation (pilot_arrays.PilotArrays checks, for X) and packaging
# =============================================================================
def check_split(root: Path, key: str, row: dict, *, exact: int | None, minimum: int | None) -> list[dict]:
    for field, digest_field in (("file", "sha256"), ("owners", "owners_sha256"), ("documents", "documents_sha256")):
        if Path(row[field]).name != row[field] or sha256_file(root / row[field]) != row[digest_field]:
            raise ValueError(f"Array identity mismatch: {key}/{field}")
    tokens = np.memmap(root / row["file"], dtype="<u2", mode="r")
    owners = np.memmap(root / row["owners"], dtype="<u4", mode="r")
    docs = read_json(root / row["documents"])["documents"]
    if len(tokens) != row["tokens"] or len(owners) != len(tokens) or int(tokens.max()) >= VOCAB:
        raise ValueError(f"Misaligned or invalid token arrays: {key}")
    if exact is not None and len(tokens) != exact or minimum is not None and len(tokens) < minimum:
        raise ValueError(f"Quota not met: {key}")
    if key.endswith("train") and (len(tokens) - 1) % CONTEXT:
        raise ValueError(f"Training array is not whole windows: {key}")
    offset = 0
    for index, doc in enumerate(docs):
        count = doc["tokens"]
        if count < 1 or doc["token_start"] != offset or not np.all(owners[offset:offset + count] == index):
            raise ValueError(f"Document attribution does not reconstruct packed positions: {key}")
        if not doc["truncated"] and tokens[offset + count - 1] != EOS:
            raise ValueError(f"Missing document EOS: {key}")
        offset += count
    if offset != len(tokens):
        raise ValueError(f"Unattributed packed tokens: {key}")
    return docs


def verify(prepared: Path, pilot_online: Path, quotas=X_QUOTAS) -> dict:
    online, orders, sealed = prepared / "s2online", prepared / "s2orders", prepared / "reserved-test"
    manifest = read_json(online / "manifest.json")
    if manifest.get("schema") != S2_ARRAYS_SCHEMA or set(manifest["splits"]) != {"x_train", "x_dev"}:
        raise ValueError("Online arrays must be exactly x_train and x_dev")
    if manifest["pilot_array_manifest_sha256"] != sha256_file(pilot_online / "manifest.json"):
        raise ValueError("Arrays were deduplicated against a different pilot corpus")
    if manifest["tokenizer_hashes"]["id_to_bytes_sha256"] != PILOT_ID_TO_BYTES_SHA256:
        raise ValueError("Tokenizer differs from the pilot's")
    if sha256_file(online / "audit.json") != manifest["dedup_audit_sha256"]:
        raise ValueError("Duplicate audit receipt mismatch")
    audit = read_json(online / "audit.json")
    if audit["missed_candidate_pairs_found"] or audit["frozen_pairs_found"] or audit["missed_candidate_pairs_checked"] < 1:
        raise ValueError("Missing/failed duplicate audit")
    selection = read_json(online / "selection.json")
    if sha256_file(online / "selection.json") != manifest["selection"]["selection_sha256"] \
            or selection["ranking"][manifest["selection"]["rank"]] != manifest["domain"]["id"]:
        raise ValueError("Selection receipt mismatch")
    docs = {"x_train": check_split(online, "x_train", manifest["splits"]["x_train"], exact=None,
                                   minimum=quota_tokens("x_train", quotas)),
            "x_dev": check_split(online, "x_dev", manifest["splits"]["x_dev"], exact=quotas["x_dev"], minimum=None)}
    if sealed.exists():
        sealed_manifest = read_json(sealed / "manifest.json")
        if sha256_file(sealed / "manifest.json") != manifest["reserved_test_sealed"]["manifest_sha256"]:
            raise ValueError("Reserved-test manifest mismatch")
        docs["x_test"] = check_split(sealed, "x_test", sealed_manifest["splits"]["x_test"],
                                     exact=quotas["x_test"], minimum=None)
    pilot_manifest = read_json(pilot_online / "manifest.json")
    frozen = {doc["content_sha256"] for key in ("web_train", "python_train", "web_dev", "python_dev")
              for doc in read_json(pilot_online / pilot_manifest["splits"][key]["documents"])["documents"]}
    seen = {}
    for key, rows in docs.items():
        for doc in rows:
            digest = doc["content_sha256"]
            if digest in frozen:
                raise ValueError("A document duplicates the pilot's frozen corpus")
            if digest in seen:
                raise ValueError(f"Exact document duplication: {seen[digest]} / {key}")
            seen[digest] = key
    order_manifest = read_json(orders / "manifest.json")
    if order_manifest["x_array_manifest_sha256"] != sha256_file(online / "manifest.json") \
            or order_manifest["pilot_array_manifest_sha256"] != manifest["pilot_array_manifest_sha256"]:
        raise ValueError("Orders belong to different arrays")
    windows = order_manifest["windows"]
    for seed in SEEDS:
        for key, row in order_manifest["seeds"][str(seed)].items():
            values = np.load(orders / row["file"], allow_pickle=False)
            if sha256_file(orders / row["file"]) != row["sha256"]:
                raise ValueError(f"Order hash mismatch {row['file']}")
            if key == "rare":
                if values.shape != (VOCAB,) or values.dtype != bool:
                    raise ValueError("Invalid rare flags")
                continue
            if len(np.unique(values)) != len(values) or values.max() >= windows[key]:
                raise ValueError(f"Invalid order {row['file']}: repeated or out-of-range windows")
    result = {"status": "accepted", "domain": manifest["domain"]["id"],
              "documents": {k: len(v) for k, v in docs.items()},
              "tokens": {k: manifest["splits"][k]["tokens"] for k in manifest["splits"]},
              "online_manifest_sha256": sha256_file(online / "manifest.json"),
              "orders_manifest_sha256": sha256_file(orders / "manifest.json")}
    json_write(prepared / "acceptance.json", result)
    return result


def package(prepared: Path, output: Path, dataset_id: str) -> Path:
    """Folder for `kaggle datasets create -p OUTPUT --dir-mode zip`: online arrays, orders, upstream evidence."""
    if output.exists():
        raise FileExistsError(f"{output} exists; choose a new folder")
    if read_json(prepared / "acceptance.json").get("status") != "accepted":
        raise ValueError("Run `verify` first")
    output.mkdir(parents=True)
    for name in ("s2online", "s2orders", "s2upstream"):
        shutil.copytree(prepared / name, output / name)
    if any(output.rglob("x_test*")):
        raise RuntimeError("Reserved test must never enter the training dataset")
    json_write(output / "dataset-metadata.json", {
        "title": dataset_id.split("/", 1)[1], "id": dataset_id, "licenses": [{"name": "other"}],
        "description": "Study 2 (domain-shift forgetting): GPT-2 token arrays of the selected domain (train/dev) and "
                       "training orders. Derived from ODC-BY sources; see s2upstream/. Reserved test excluded."})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("probe-windows")
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--cache", type=Path, default=Path("/tmp/s2probe-cache"))
    prep = sub.add_parser("prepare")
    prep.add_argument("--inputs", type=Path, default=Path("/kaggle/input"))
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--cache", type=Path, required=True)
    prep.add_argument("--rank", type=int, default=0, help="0 = the selection; 1 = the pre-registered fallback only")
    check = sub.add_parser("verify")
    check.add_argument("--prepared", type=Path, required=True)
    check.add_argument("--pilot-online", type=Path, default=None)
    check.add_argument("--inputs", type=Path, default=Path("/kaggle/input"))
    pack = sub.add_parser("package")
    pack.add_argument("--prepared", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--dataset-id", default="danny00/study2-online-inputs")
    args = parser.parse_args()
    if args.command == "probe-windows":
        result = build_probe_windows(args.output, args.cache)
        result = {cid: {k: v for k, v in info.items() if k != "documents"} for cid, info in result["candidates"].items()}
    elif args.command == "prepare":
        paths = locate(args.inputs)
        manifest = prepare(paths["selection"], paths["pilot_online"], paths["pilot_orders"], args.output, args.cache,
                           args.rank)
        result = verify(args.output, paths["pilot_online"]) | {"raw_tokens": manifest["raw_tokens_before_dedup"]}
    elif args.command == "verify":
        result = verify(args.prepared, args.pilot_online or locate(args.inputs)["pilot_online"])
    else:
        result = {"dataset_folder": str(package(args.prepared, args.output, args.dataset_id))}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    sys.exit(main())
