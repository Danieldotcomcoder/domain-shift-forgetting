"""Verified online-only arrays, immutable orders and complete development statistics."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .local_corpus import TokenStream, sha256
from .pilot_control import write_json, read_json
from .protocol import PREFIX_END, CONTINUATION_UPDATES, SEEDS
from .evaluation.statistics import EvaluationSums


class PilotArrays:
    def __init__(self, root: Path, *, strict: bool = True):
        self.root = root.resolve()
        self.manifest = read_json(root / "manifest.json")
        if self.manifest.get("schema") != "pilot-arrays-v1":
            raise ValueError("Require original-protocol arrays; local/proxy corpora are rejected")
        if strict:
            from .pilot_data import REPOS
            for domain, repo in REPOS.items():
                source = self.manifest["sources"][domain]
                if source.get("repo") != repo or len(source.get("revision", "")) != 40:
                    raise ValueError("Original source revision is missing or substituted")
            if sha256(root / "audit.json") != self.manifest["dedup_audit_sha256"]:
                raise ValueError("Duplicate audit receipt mismatch")
            audit = read_json(root / "audit.json")
            if audit["missed_candidate_pairs_found"] != 0 or audit["missed_candidate_pairs_checked"] < 1:
                raise ValueError("Missing/failed duplicate audit")
            sealed = self.manifest["reserved_test_sealed"]
            if sealed["labels_per_domain"] != 8388608 or sealed["online_path_included"] is not False:
                raise ValueError("Reserved test isolation is not established")
        self.identity = sha256(root / "manifest.json")
        self.streams, self.owners, self.docs = {}, {}, {}
        expected = {f"{domain}_{split}" for domain in ("web", "python") for split in ("train", "dev")}
        if set(self.manifest["splits"]) != expected:
            raise ValueError("Only the four original online train/dev splits are allowed; no reserved test references")
        for key, row in self.manifest["splits"].items():
            for field, digest_field in (("file", "sha256"), ("owners", "owners_sha256"), ("documents", "documents_sha256")):
                path = root / row[field]
                if Path(row[field]).name != row[field] or sha256(path) != row[digest_field]:
                    raise ValueError(f"Array identity mismatch: {key}/{field}")
            stream = TokenStream(root / row["file"], 512)
            owners = np.memmap(root / row["owners"], dtype="<u4", mode="r")
            docs = read_json(root / row["documents"])["documents"]
            if len(stream.tokens) != row["tokens"] or len(owners) != len(stream.tokens) or int(stream.tokens.max()) >= 50257:
                raise ValueError("Misaligned or invalid token arrays")
            if not docs or int(owners.max()) >= len(docs):
                raise ValueError("Invalid document owners")
            offset = 0
            for index, doc in enumerate(docs):
                count = doc["tokens"]
                if count < 1 or doc["token_start"] != offset or not np.all(owners[offset:offset + count] == index):
                    raise ValueError("Document attribution does not reconstruct packed positions")
                if not doc["truncated"] and stream.tokens[offset + count - 1] != 50256:
                    raise ValueError("Missing document EOS")
                offset += count
            if offset != len(stream.tokens):
                raise ValueError("Unattributed packed tokens")
            if strict:
                minimum = 260_000_000 if key == "web_train" else 110_000_000 if key == "python_train" else 2_097_153
                if len(stream.tokens) < minimum or (key.endswith("dev") and len(stream.tokens) != minimum):
                    raise ValueError(f"Original data quota not met: {key}")
            self.streams[key], self.owners[key], self.docs[key] = stream, owners, docs
        if sha256(root / "classes.json") != self.manifest["classes_sha256"]:
            raise ValueError("Token class identity mismatch")
        self.classes = np.array(read_json(root / "classes.json")["classes"])
        if self.classes.shape != (50257,) or not set(self.classes) <= set("WAPX"):
            raise ValueError("Invalid tokenizer class table")
        seen, alias_splits = {}, {}
        for key, docs in self.docs.items():
            for doc in docs:
                digest = doc["content_sha256"]
                if digest in seen:
                    raise ValueError("Exact document leakage/duplication in materialized corpus")
                seen[digest] = key
                if key.startswith("python"):
                    for alias in doc["aliases"]:
                        if alias in alias_splits and alias_splits[alias] != key:
                            raise ValueError("Repository leakage across online splits")
                        alias_splits[alias] = key

    def make_orders(self, output: Path) -> dict:
        output.mkdir(parents=True, exist_ok=False)
        manifest = {"array_manifest_sha256": self.identity, "seeds": {}, "probe_rule": "last 256 training windows, forward-only"}
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            web = rng.permutation(self.streams["web_train"].windows)
            code = rng.permutation(self.streams["python_train"].windows)
            needed_web = (PREFIX_END + CONTINUATION_UPDATES) * 32
            needed_code = CONTINUATION_UPDATES * 32
            if len(web) < needed_web or len(code) < needed_code:
                raise ValueError("Insufficient unique windows; replacement sampling is forbidden")
            paths = {}
            for name, values in (("web", web[:needed_web]), ("python", code[:needed_code])):
                path = output / f"{seed}-{name}.npy"
                np.save(path, values.astype("<u8"), allow_pickle=False)
                paths[name] = {"file": path.name, "sha256": sha256(path)}
            # Unique prefix positions, counting shared adjacent context only once.
            counts = np.zeros(50257, dtype=np.int64)
            selected = np.sort(web[:PREFIX_END * 32])
            previous = -2
            for start in range(0, len(selected), 1024):
                indices = selected[start:start + 1024]
                rows = self.streams["web_train"].take(indices)
                counts += np.bincount(rows[:, 1:].ravel(), minlength=50257)
                include = indices != np.concatenate(([previous], indices[:-1])) + 1
                counts += np.bincount(rows[include, 0], minlength=50257)
                previous = indices[-1]
            rare_path = output / f"{seed}-rare.npy"
            np.save(rare_path, counts < 100, allow_pickle=False)
            paths["rare"] = {"file": rare_path.name, "sha256": sha256(rare_path)}
            manifest["seeds"][str(seed)] = paths
        write_json(output / "manifest.json", manifest)
        return manifest


def load_orders(root: Path, arrays: PilotArrays, seed: int) -> dict:
    manifest = read_json(root / "manifest.json")
    if manifest["array_manifest_sha256"] != arrays.identity:
        raise ValueError("Orders belong to different arrays")
    result = {}
    for key, row in manifest["seeds"][str(seed)].items():
        if Path(row["file"]).name != row["file"] or sha256(root / row["file"]) != row["sha256"]:
            raise ValueError("Order identity mismatch")
        result[key] = np.load(root / row["file"], mmap_mode="r", allow_pickle=False)
    return result


@torch.no_grad()
def evaluate(model, arrays: PilotArrays, domain: str, *, role: str, rare: np.ndarray,
             microbatch: int, precision: str, clock=None) -> dict:
    if domain not in ("web", "python") or role not in ("full", "quick"):
        raise ValueError("Development-only evaluator")
    key = f"{domain}_dev"
    stream = arrays.streams[key]
    # The first 512 frozen full-dev windows are the fixed quick subset.
    windows = 4096 if role == "full" else 512
    if stream.windows < windows:
        raise ValueError("Development quota incomplete")
    device = next(model.parameters()).device
    model.eval()
    result = EvaluationSums()
    for start in range(0, windows, microbatch):
        if clock is not None and clock.must_stop(10):
            raise TimeoutError("Pause before partial evaluation consumes checkpoint/export reserve")
        indices = np.arange(start, min(windows, start + microbatch))
        rows = torch.from_numpy(stream.take(indices)).to(device)
        with torch.autocast(device.type, dtype=torch.float16 if precision == "fp16" else torch.bfloat16,
                            enabled=precision != "fp32"):
            logits, _ = model(rows[:, :-1], collect=False, auxiliary=False)
            losses = F.cross_entropy(logits.float().reshape(-1, 50257), rows[:, 1:].reshape(-1), reduction="none")
        labels = rows[:, 1:].cpu().numpy().ravel()
        offsets = indices[:, None] * 512 + np.arange(1, 513)
        owners = arrays.owners[key][offsets].ravel()
        doc_ids = [arrays.docs[key][int(i)]["id"] for i in owners]
        result.add(losses.cpu().tolist(), doc_ids, arrays.classes[labels].tolist(), rare[labels].tolist())
    result.check_reconstruction()
    if result.total.count != windows * 512:
        raise ValueError("Incomplete evaluation")
    return {"domain": domain, "role": role, "array_sha256": arrays.manifest["splits"][key]["sha256"],
            "ce": result.total.mean, "non_w_ce": result.code_ce(), "ap_ce": result.code_ce(alphanumeric_punctuation_only=True),
            "statistics": asdict(result)}
