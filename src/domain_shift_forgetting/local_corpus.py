"""Prepare larger, GPT-2-tokenized learning data; never touches pilot datasets."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np


SOURCES = {
    "web": {"repo": "Salesforce/wikitext", "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
            "directory": "wikitext-103-raw-v1", "description": "Wikipedia prose, not general web",
            "license_note": "See the preserved upstream WikiText card and attribution."},
    "python": {"repo": "code-search-net/code_search_net", "revision": "bd0cf261e357a3eb5c8fba490d23ec1a1cd59555",
               "directory": "python", "description": "CodeSearchNet Python functions",
               "license_note": "Source repository licenses vary; preserve code URLs and consult upstream licensing."},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def remote_json(url: str) -> dict | list:
    with urlopen(Request(url, headers={"User-Agent": "domain-shift-local-lab"}), timeout=60) as response:
        return json.load(response)


def download(url: str, path: Path, expected_sha: str | None = None) -> Path:
    """Resume partial transfers; publish only after length/hash validation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if expected_sha and sha256(path) != expected_sha:
            raise ValueError(f"Cached source checksum mismatch: {path}")
        return path
    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(5):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "domain-shift-local-lab"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with urlopen(Request(url, headers=headers), timeout=120) as response:
                resumed = response.status == 206
                if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Server returned an unexpected byte range")
                expected_size = response.headers.get("Content-Length")
                received = 0
                last_report = time.monotonic()
                with partial.open("ab" if resumed else "wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        received += len(chunk)
                        if time.monotonic() - last_report > 10:
                            print(f"Downloading {path.name}: {output.tell() / 1e6:.1f} MB", flush=True)
                            last_report = time.monotonic()
                if expected_size and received != int(expected_size):
                    raise OSError("Incomplete HTTP response")
            if expected_sha and sha256(partial) != expected_sha:
                partial.unlink()
                raise ValueError("Source checksum mismatch; retrying download")
            partial.replace(path)
            return path
        except (OSError, ValueError) as exc:
            if getattr(exc, "code", None) == 416 and partial.exists():
                # The transfer may have completed just before a connection reset.
                if expected_sha and sha256(partial) == expected_sha:
                    partial.replace(path)
                    return path
                partial.unlink()
            if attempt == 4:
                raise
            print(f"Download retry: {exc}", flush=True)
            time.sleep(2 ** attempt)
    raise RuntimeError("Download did not complete")


def source_files(source: dict) -> list[dict]:
    url = (f"https://huggingface.co/api/datasets/{source['repo']}/tree/"
           f"{source['revision']}/{source['directory']}")
    entries = remote_json(url)
    # No upstream test file is selected or downloaded.
    return sorted([row for row in entries if row["path"].endswith(".parquet")
                   and Path(row["path"]).name.startswith(("train-", "validation-"))], key=lambda x: x["path"])


def rows(source: dict, files: list[dict], split: str, cache: Path, receipts: list[dict]):
    import pyarrow.parquet as pq
    for entry in files:
        if not Path(entry["path"]).name.startswith(split + "-"):
            continue
        url = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{quote(entry['path'])}"
        expected = entry.get("lfs", {}).get("oid")
        local = download(url, cache / source["repo"].replace("/", "--") / source["revision"] / entry["path"], expected)
        receipts.append({"url": url, "sha256": sha256(local), "bytes": local.stat().st_size})
        columns = ["text"] if source["directory"].startswith("wikitext") else [
            "repository_name", "whole_func_string", "func_code_url"]
        for batch in pq.ParquetFile(local).iter_batches(batch_size=256, columns=columns):
            yield from batch.to_pylist()


def wiki_articles(records):
    """Join lines into articles before deduplication/tokenization/split checks."""
    title, lines = None, []
    for row in records:
        text = row["text"]
        if re.match(r"^= [^=](?:.*[^=])? =$", text.strip()):
            if title and lines:
                yield {"text": "".join(lines), "group": title, "source": title}
            title, lines = text.strip(), [text]
        elif title:
            lines.append(text)
    if title and lines:
        yield {"text": "".join(lines), "group": title, "source": title}


def code_documents(records):
    for row in records:
        yield {"text": row["whole_func_string"], "group": row["repository_name"].casefold(), "source": row["func_code_url"]}


def write_split(documents, path: Path, cap: int, encoder, seen: set[str], blocked_groups: set[str]) -> tuple[dict, set[str]]:
    """Dev precedes train. Keep document provenance; cap may truncate the last doc."""
    tokens = count = duplicates = blocked = 0
    groups: set[str] = set()
    with path.open("wb") as stream, path.with_suffix(".documents.jsonl").open("w", encoding="utf-8") as records:
        for doc in documents:
            if doc["group"] in blocked_groups:
                blocked += 1
                continue
            digest = hashlib.sha256(doc["text"].encode("utf-8")).hexdigest()
            if digest in seen or not doc["text"].strip():
                duplicates += 1
                continue
            ids = encoder.encode_ordinary(doc["text"]) + [encoder.eot_token]
            kept = ids[:cap - tokens]
            np.asarray(kept, dtype="<u2").tofile(stream)
            records.write(json.dumps({"sha256": digest, "group": doc["group"], "source": doc["source"],
                                      "token_start": tokens, "tokens": len(kept), "truncated": len(kept) < len(ids)}) + "\n")
            seen.add(digest)
            groups.add(doc["group"])
            tokens += len(kept)
            count += 1
            if tokens >= cap:
                break
    if tokens < min(1025, cap):
        raise ValueError(f"Insufficient usable data in {path.name}")
    return {"tokens": tokens, "documents": count, "groups": len(groups), "exact_duplicates_skipped": duplicates,
            "heldout_group_rows_skipped": blocked, "sha256": sha256(path), "file": path.name}, groups


def prepare(output: Path, cache: Path, train_tokens: int, dev_tokens: int) -> None:
    import tiktoken
    if train_tokens < 1025 or dev_tokens < 1025:
        raise ValueError("Each token cap must be at least 1025")
    if (output / "manifest.json").exists():
        raise FileExistsError("Prepared corpus already exists; use a new output directory")
    settings = {"sources": SOURCES, "train_tokens": train_tokens, "dev_tokens": dev_tokens}
    if output.exists() and any(output.iterdir()):
        marker = output / "preparing.json"
        if not marker.exists() or json.loads(marker.read_text()) != settings:
            raise ValueError("Output is not a matching unfinished preparation")
    output.mkdir(parents=True, exist_ok=True)
    json_write(output / "preparing.json", settings)
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str((cache / "tokenizer").resolve()))
    encoder = tiktoken.get_encoding("gpt2")
    if encoder.n_vocab != 50257 or encoder.eot_token != 50256:
        raise ValueError("Unexpected GPT-2 encoding")
    manifest = {"schema": 1, "purpose": "larger local learning; not protocol-v3 data",
                "tokenizer": "gpt2", "vocab_size": 50257, "eos": 50256, "dtype": "<u2",
                "tiktoken_version": importlib.metadata.version("tiktoken"),
                "sources": SOURCES, "splits": {}, "downloads": []}
    catalogs = {domain: source_files(source) for domain, source in SOURCES.items()}
    for domain, source in SOURCES.items():
        download(f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/README.md",
                 output / f"{domain}-upstream-card.md")
    seen: set[str] = set()
    heldout: dict[str, set[str]] = {}
    # Give development data precedence across BOTH domains.
    for split in ("dev", "train"):
        for domain, source in SOURCES.items():
            records = rows(source, catalogs[domain], "validation" if split == "dev" else "train", cache, manifest["downloads"])
            docs = wiki_articles(records) if domain == "web" else code_documents(records)
            key = f"{domain}_{split}"
            info, groups = write_split(docs, output / f"{key}.bin", dev_tokens if split == "dev" else train_tokens,
                                       encoder, seen, heldout.get(domain, set()) if split == "train" else set())
            manifest["splits"][key] = info
            if split == "dev":
                heldout[domain] = groups
            print(f"{key}: {info['tokens']:,} tokens, {info['documents']:,} documents", flush=True)
    # Publication of the manifest is the completion marker.
    json_write(output / "manifest.json", manifest)
    print(f"Prepared {output.resolve()}")


class TokenStream:
    def __init__(self, path: Path, context: int):
        self.tokens = np.memmap(path, dtype="<u2", mode="r")
        self.context = context
        self.windows = (len(self.tokens) - 1) // context
        if self.windows < 1:
            raise ValueError(f"Not enough tokens in {path}")

    def take(self, indices: np.ndarray) -> np.ndarray:
        offsets = indices[:, None] * self.context + np.arange(self.context + 1)
        return np.asarray(self.tokens[offsets], dtype=np.int64)


def load_corpus(root: Path, context: int) -> tuple[dict[str, TokenStream], str]:
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != 1 or manifest.get("tokenizer") != "gpt2" or manifest.get("dtype") != "<u2":
        raise ValueError("Require a completed GPT-2 local corpus")
    result = {}
    for key in ("web_train", "web_dev", "python_train", "python_dev"):
        info = manifest["splits"][key]
        file = root / f"{key}.bin"
        if file.stat().st_size != info["tokens"] * 2 or sha256(file) != info["sha256"]:
            raise ValueError(f"Corpus integrity check failed: {key}")
        result[key] = TokenStream(file, context)
        if int(result[key].tokens.max()) >= 50257:
            raise ValueError("Out-of-vocabulary token in corpus")
    return result, sha256(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("data/local-large"))
    p.add_argument("--cache", type=Path, default=Path("data/local-downloads"))
    p.add_argument("--train-tokens", type=int, default=50_000_000, help="Per domain")
    p.add_argument("--dev-tokens", type=int, default=131_073, help="Per domain")
    args = p.parse_args()
    prepare(args.output, args.cache, args.train_tokens, args.dev_tokens)


if __name__ == "__main__":
    main()
