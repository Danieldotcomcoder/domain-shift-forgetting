"""Original-dataset access inspection and deterministic preparation (CPU only).

Preparation, including reserved-test isolation, precedes all scientific training.
Never substitutes WikiText/CodeSearchNet or repeats documents to meet quotas.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import sqlite3
import struct
import unicodedata

import numpy as np

from .data.classes import token_class
from .data.splits import RepositoryDocument, repository_groups, python_split, c4_validation_split
from .local_corpus import json_write, sha256

SAMPLING_SEED = 20260911
REPOS = {"web": "allenai/c4", "python": "bigcode/the-stack-dedup"}
QUOTAS = {"web_train": 260_000_000, "python_train": 110_000_000,
          "web_dev": 2_097_153, "python_dev": 2_097_153,
          "web_test": 8_388_609, "python_test": 8_388_609}


def inspect_access(output: Path) -> dict:
    """Read one authorized example per source; save schema/types, never raw text/token."""
    from datasets import load_dataset
    from huggingface_hub import HfApi
    api = HfApi()  # Reads HF_TOKEN from the environment; never writes it to reports.
    output.mkdir(parents=True, exist_ok=True)
    result = {"schema": 1, "status": "passed", "sources": {},
              "notice": "Access/schema evidence only, not completed corpus validation"}
    for domain, repo in REPOS.items():
        try:
            revision = api.dataset_info(repo).sha
            paths = api.list_repo_files(repo, repo_type="dataset", revision=revision)
            selected = sorted(p for p in paths if
                (p.startswith("en/c4-train.") and p.endswith(".json.gz") if domain == "web"
                 else p.startswith("data/python/") and p.endswith(".parquet")))
            if not selected:
                raise ValueError("No expected source shards at this revision")
            row = next(iter(load_dataset("json" if domain == "web" else "parquet",
                data_files=f"hf://datasets/{repo}@{revision}/{selected[0]}",
                split="train", streaming=True)))
            required = "text" if domain == "web" else "content"
            if not isinstance(row.get(required), str):
                raise ValueError("Unexpected text schema")
            aliases = aliases_from_row(row) if domain == "python" else []
            if domain == "python" and not aliases:
                raise ValueError("Sample lacks usable repository aliases")
            result["sources"][domain] = {"repo": repo, "revision": revision, "sample_shard": selected[0],
                "schema": {k: type(v).__name__ for k, v in row.items()},
                "sample_content_sha256": hashlib.sha256(row[required].encode()).hexdigest(),
                "alias_fields": [k for k in row if k.endswith("_repo_name") or k == "repository_name"],
                "license_fields": [k for k in row if "license" in k], "status": "passed"}
        except Exception as exc:
            # Some HTTP errors embed signed URLs; expose only the exception class.
            result["status"] = "blocked"
            result["sources"][domain] = {"repo": repo, "status": "blocked", "error_type": type(exc).__name__,
                "action": "Check HF_TOKEN, dataset access approval, current usable version and network."}
    json_write(output / "access.json", result)
    return result


def aliases_from_row(row: dict) -> tuple[str, ...]:
    aliases = []
    for key, value in row.items():
        if key == "repository_name" or key.endswith("_repo_name"):
            if isinstance(value, str) and value.strip():
                aliases.append(value.strip().removeprefix("https://github.com/").removesuffix(".git").casefold())
    return tuple(sorted(set(aliases)))


def _shingles(blob: bytes) -> set[bytes]:
    tokens = np.frombuffer(blob, dtype="<u2")
    return {tokens[i:i + 5].tobytes() for i in range(max(0, len(tokens) - 4))}


def filter_pool(db: sqlite3.Connection, audit_path: Path) -> dict[str, list[str]]:
    """Rebuild repository components and dedup after every pool expansion."""
    from datasketch import MinHash, MinHashLSH
    rows = db.execute("SELECT id,domain,upstream,content_hash,metadata FROM docs ORDER BY id").fetchall()
    repos = [RepositoryDocument(row[0], tuple(json.loads(row[4])["aliases"])) for row in rows if row[1] == "python"]
    groups = repository_groups(repos)
    classified = []
    for doc_id, domain, upstream, content_hash, meta in rows:
        split = python_split(groups[doc_id]) if domain == "python" else (
            "train" if upstream == "train" else c4_validation_split(content_hash))
        metadata = json.loads(meta)
        metadata.update(split=split, group_sha256=groups.get(doc_id))
        db.execute("UPDATE docs SET metadata=? WHERE id=?", (json.dumps(metadata), doc_id))
        classified.append((doc_id, domain, split, content_hash))
    priority = {"test": 0, "dev": 1, "train": 2}
    classified.sort(key=lambda r: (priority[r[2]], r[0]))
    index = MinHashLSH(num_perm=128, params=(32, 4))
    seen, retained = {}, {key: [] for key in QUOTAS}
    short, exact, near = 0, 0, 0
    with audit_path.open("w", encoding="utf-8") as log:
        for doc_id, domain, split, digest in classified:
            if digest in seen:
                exact += 1
                log.write(json.dumps({"removed": doc_id, "retained": seen[digest], "reason": "exact"}) + "\n")
                continue
            blob = db.execute("SELECT tokens FROM docs WHERE id=?", (doc_id,)).fetchone()[0]
            shingles = _shingles(blob)
            signature = MinHash(num_perm=128, seed=SAMPLING_SEED)
            duplicate = None
            if shingles:
                signature.update_batch(sorted(shingles))
                for candidate in sorted(index.query(signature)):
                    other = _shingles(db.execute("SELECT tokens FROM docs WHERE id=?", (candidate,)).fetchone()[0])
                    if len(shingles & other) / len(shingles | other) >= 0.85:
                        duplicate = candidate
                        break
            else:
                short += 1
            if duplicate:
                near += 1
                log.write(json.dumps({"removed": doc_id, "retained": duplicate, "reason": "near_jaccard_ge_0.85"}) + "\n")
                continue
            seen[digest] = doc_id
            retained[f"{domain}_{split}"].append(doc_id)
            if shingles:
                index.insert(doc_id, signature)
    # A fixed random audit of retained cross-split pairs, independent of outcomes.
    flat = [(key, doc) for key, docs in retained.items() for doc in docs]
    rng = random.Random(SAMPLING_SEED)
    checked = misses = 0
    for _ in range(min(10_000, len(flat) * 4)):
        a, b = rng.sample(flat, 2) if len(flat) >= 2 else (None, None)
        if a is None or a[0].split("_")[1] == b[0].split("_")[1]:
            continue
        sa, sb = (_shingles(db.execute("SELECT tokens FROM docs WHERE id=?", (x[1],)).fetchone()[0]) for x in (a, b))
        if sa and sb:
            checked += 1
            misses += int(len(sa & sb) / len(sa | sb) >= 0.85)
    json_write(audit_path.with_suffix(".summary.json"), {"exact_removed": exact, "near_removed": near,
        "short_shingle_documents": short, "missed_candidate_pairs_checked": checked,
        "missed_candidate_pairs_found": misses, "repository_groups": len(set(groups.values())),
        "minhash_permutations": 128, "seed": SAMPLING_SEED, "lsh_bands": 32, "lsh_rows": 4})
    if misses:
        raise ValueError("Missed-candidate audit found leakage; investigate before freezing")
    return retained


def materialize(db, retained: dict, output: Path, tokenizer_hashes: dict, sources: dict,
                quotas: dict = QUOTAS) -> dict:
    """Whole packed windows; owners align one-to-one with tokens, including EOS."""
    online, sealed = output / "online", output / "reserved-test"
    online.mkdir(parents=True, exist_ok=True)
    sealed.mkdir(parents=True, exist_ok=True)
    manifests = {p: {"schema": "pilot-arrays-v1", "sources": sources, "splits": {},
                      "tokenizer_hashes": tokenizer_hashes, "sampling_seed": SAMPLING_SEED} for p in (online, sealed)}
    for key, ids in retained.items():
        domain, split = key.split("_")
        target = sealed if split == "test" else online
        # A fixed content-ID ordering is shared across seeds before permutation.
        ids = sorted(ids, key=lambda value: hashlib.sha256(f"{SAMPLING_SEED}:{value}".encode()).hexdigest())
        token_path, owner_path = target / f"{key}.bin", target / f"{key}.owners.bin"
        wanted = quotas[key]
        # Reserve training tokens as complete windows + the initial context.
        wanted = ((wanted - 1 + 511) // 512) * 512 + 1 if split == "train" else wanted
        count, docs = 0, []
        with token_path.open("wb") as tokens, owner_path.open("wb") as owners:
            for doc_id in ids:
                blob, meta = db.execute("SELECT tokens,metadata FROM docs WHERE id=?", (doc_id,)).fetchone()
                values = np.concatenate((np.frombuffer(blob, dtype="<u2"), np.array([50256], dtype="<u2")))
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
        manifests[target]["splits"][key] = {"tokens": count, "labels": count - 1,
            "file": token_path.name, "sha256": sha256(token_path), "owners": owner_path.name,
            "owners_sha256": sha256(owner_path), "documents": docs_path.name,
            "documents_sha256": sha256(docs_path), "initial_context_tokens": 1}
    for path, manifest in manifests.items():
        json_write(path / "manifest.json", manifest)
    return manifests[online]


def prepare(access: Path, output: Path, cache: Path, max_shards: int = 64, *, resume: bool = False) -> dict:
    """Bounded deterministic expansion; reports a deficit instead of repeating data."""
    from datasets import load_dataset
    from huggingface_hub import HfApi
    import tiktoken
    evidence = json.loads(access.read_text())
    if evidence.get("status") != "passed":
        raise ValueError("Run access inspection successfully first")
    settings = {"sources": evidence["sources"], "code_sha256": sha256(Path(__file__)), "seed": SAMPLING_SEED}
    if output.exists() and not resume:
        raise FileExistsError("Choose a new preparation directory; retain prior audit evidence")
    if resume and (not (output / "preparation-settings.json").exists() or
                   json.loads((output / "preparation-settings.json").read_text()) != settings):
        raise ValueError("Preparation source/code identity changed; use a new directory")
    final_manifest = output / "online/manifest.json"
    if final_manifest.exists() and json.loads(final_manifest.read_text()).get("preparation_status"):
        raise FileExistsError("Preparation already completed")
    output.mkdir(parents=True, exist_ok=True)
    json_write(output / "preparation-settings.json", settings)
    cache.mkdir(parents=True, exist_ok=True)
    encoder = tiktoken.get_encoding("gpt2")
    classes = [token_class(i, encoder.decode_single_token_bytes(i)) for i in range(50257)]
    byte_map = hashlib.sha256(b"".join(struct.pack("<I", len(encoder.decode_single_token_bytes(i))) +
                                     encoder.decode_single_token_bytes(i) for i in range(50257))).hexdigest()
    tokenizer_hashes = {"encoding": "gpt2", "id_to_bytes_sha256": byte_map,
                        "tiktoken": importlib.metadata.version("tiktoken"), "unicode": unicodedata.unidata_version}
    api = HfApi()
    db = sqlite3.connect(output / "candidate-pool.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS docs(id TEXT PRIMARY KEY, domain TEXT, upstream TEXT, content_hash TEXT, tokens BLOB, metadata TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS shards(path TEXT PRIMARY KEY, receipt TEXT)")
    consumed = [json.loads(row[0]) for row in db.execute("SELECT receipt FROM shards ORDER BY path")]
    consumed_paths = {row["shard"] for row in consumed}
    files = {}
    for domain, source in evidence["sources"].items():
        all_files = api.list_repo_files(source["repo"], repo_type="dataset", revision=source["revision"])
        for upstream in (("train", "validation") if domain == "web" else ("train",)):
            selected = [p for p in all_files if (p.startswith(f"en/c4-{upstream}.") and p.endswith(".json.gz")
                        if domain == "web" else p.startswith("data/python/") and p.endswith(".parquet"))]
            files[domain, upstream] = sorted(selected, key=lambda p: hashlib.sha256(f"{SAMPLING_SEED}:{p}".encode()).hexdigest())
    totals = Counter({(a, b): n for a, b, n in db.execute("SELECT domain,upstream,sum(length(tokens)/2+1) FROM docs GROUP BY domain,upstream")})
    resume_round = max((i for values in files.values() for i, name in enumerate(values) if name in consumed_paths), default=0)
    retained = None
    try:
        for round_index in range(max_shards):
            advanced = False
            for (domain, upstream), candidates in files.items():
                # Replay the same source rounds after interruption. Completed shards
                # are atomic receipts; never advance an already completed source twice.
                shard = candidates[round_index] if round_index < len(candidates) else None
                if shard is None or shard in consumed_paths:
                    continue
                advanced = True
                source = evidence["sources"][domain]
                uri = f"hf://datasets/{source['repo']}@{source['revision']}/{shard}"
                rows = load_dataset("json" if domain == "web" else "parquet", data_files=uri,
                                    split="train", streaming=True, cache_dir=str(cache))
                counts = Counter()
                for index, row in enumerate(rows):
                    text = row["text" if domain == "web" else "content"]
                    aliases = aliases_from_row(row) if domain == "python" else ()
                    if not text.strip() or (domain == "python" and not aliases):
                        counts["dropped_missing_text_or_provenance"] += 1
                        continue
                    tokens = np.array(encoder.encode_ordinary(text), dtype="<u2")
                    digest = hashlib.sha256(text.encode()).hexdigest()
                    doc_id = hashlib.sha256(f"{source['revision']}:{shard}:{index}".encode()).hexdigest()
                    meta = {"dataset": source["repo"], "revision": source["revision"], "shard": shard,
                            "row": index, "content_sha256": digest, "aliases": aliases, "url": row.get("url"),
                            "licenses": {k: v for k, v in row.items() if "license" in k}}
                    db.execute("INSERT INTO docs VALUES (?,?,?,?,?,?)", (doc_id, domain, upstream, digest, tokens.tobytes(), json.dumps(meta)))
                    totals[domain, upstream] += len(tokens) + 1
                    counts["kept"] += 1
                receipt = {"domain": domain, "upstream": upstream, "shard": shard, "counts": dict(counts)}
                # Rows and their completed-shard receipt commit together.
                db.execute("INSERT INTO shards VALUES (?,?)", (shard, json.dumps(receipt)))
                db.commit()
                consumed.append(receipt)
                consumed_paths.add(shard)
                json_write(output / "preparation-progress.json", {"round": round_index, "consumed_shards": consumed,
                    "raw_token_counts": {f"{a}/{b}": n for (a, b), n in totals.items()}})
            if not advanced and all(round_index >= len(values) for values in files.values()):
                break
            if round_index < resume_round:
                continue
            # Avoid expensive full-pool grouping before even raw quotas can fit.
            if totals["web", "train"] < QUOTAS["web_train"] or totals["web", "validation"] < 2 * QUOTAS["web_test"] or totals["python", "train"] < 2 * QUOTAS["python_train"]:
                continue
            retained = filter_pool(db, output / "dedup-removals.jsonl")
            available = {key: sum(db.execute("SELECT length(tokens)/2+1 FROM docs WHERE id=?", (doc,)).fetchone()[0]
                                  for doc in ids) for key, ids in retained.items()}
            if all(available[key] >= quota + (512 if key.endswith("train") else 0) for key, quota in QUOTAS.items()):
                break
        if retained is None or not all(available[key] >= quota + (512 if key.endswith("train") else 0) for key, quota in QUOTAS.items()):
            raise ValueError("Candidate pool too small within shard limit; no arrays frozen; increase --max-shards with --resume")
        manifest = materialize(db, retained, output, tokenizer_hashes, evidence["sources"])
        json_write(output / "online/classes.json", {"classes": classes, "tokenizer": tokenizer_hashes})
        # Only references to online arrays enter the training manifest.
        manifest["classes_sha256"] = sha256(output / "online/classes.json")
        shutil.copyfile(output / "dedup-removals.summary.json", output / "online/audit.json")
        manifest["dedup_audit_sha256"] = sha256(output / "online/audit.json")
        manifest["reserved_test_sealed"] = {"manifest_sha256": sha256(output / "reserved-test/manifest.json"),
            "labels_per_domain": 8388608, "online_path_included": False}
        manifest["preparation_packages"] = {name: importlib.metadata.version(name) for name in
            ("numpy", "tiktoken", "pyarrow", "huggingface_hub", "datasets", "datasketch")}
        manifest["preparation_status"] = "prepared_requires_acceptance_validation"
        json_write(output / "online/manifest.json", manifest)
        return manifest
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--output", type=Path, required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--access", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--cache", type=Path, required=True)
    prep.add_argument("--max-shards", type=int, default=64)
    prep.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "inspect":
        result = inspect_access(args.output)
        print(json.dumps(result, indent=2))
        if result["status"] != "passed":
            raise SystemExit(2)
    else:
        prepare(args.access, args.output, args.cache, args.max_shards, resume=args.resume)


if __name__ == "__main__":
    main()
