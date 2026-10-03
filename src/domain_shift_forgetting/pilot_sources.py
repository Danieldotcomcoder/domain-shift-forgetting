"""Capture actual upstream revisions/documents; do not invent source pins."""
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .local_corpus import sha256
from .pilot_control import write_json


def fetch_source(url: str) -> bytes:
    """Use Hub authentication for dataset cards; never send it to other hosts."""
    parsed = urlsplit(url)
    try:
        if parsed.hostname == "huggingface.co":
            from huggingface_hub import hf_hub_download
            parts = parsed.path.strip("/").split("/")
            if len(parts) < 6 or parts[0] != "datasets" or parts[3] != "raw":
                raise ValueError("Unexpected dataset-card URL")
            path = hf_hub_download(repo_id="/".join(parts[1:3]), repo_type="dataset",
                revision=parts[4], filename="/".join(parts[5:]), token=os.environ.get("HF_TOKEN"))
            return Path(path).read_bytes()
        with urlopen(Request(url, headers={"User-Agent": "domain-shift-research-preparation"}), timeout=60) as response:
            return response.read()
    except Exception as exc:
        status = exc.code if isinstance(exc, HTTPError) else getattr(getattr(exc, "response", None), "status_code", None)
        # Do not print signed URLs, headers, or token-bearing exception messages.
        raise RuntimeError(f"Source capture failed for {parsed.hostname}{parsed.path}: "
                           f"{type(exc).__name__}, HTTP {status if status is not None else 'unknown'}") from None


def capture(output: Path) -> dict:
    if (output / "source-evidence.json").exists():
        raise FileExistsError("Source capture already completed; preserve its evidence")
    output.mkdir(parents=True, exist_ok=True)

    ref = json.loads(fetch_source("https://api.github.com/repos/karpathy/nanoGPT/commits/master"))
    commit = ref["sha"]
    urls = {"nanogpt-LICENSE.txt": f"https://raw.githubusercontent.com/karpathy/nanoGPT/{commit}/LICENSE",
            "nanogpt-model.py.txt": f"https://raw.githubusercontent.com/karpathy/nanoGPT/{commit}/model.py",
            "tapernorm-v1.html": "https://arxiv.org/html/2602.10408v1",
            "stack-card.md": "https://huggingface.co/datasets/bigcode/the-stack-dedup/raw/main/README.md",
            "c4-card.md": "https://huggingface.co/datasets/allenai/c4/raw/main/README.md"}
    entries = {}
    for name, url in urls.items():
        path = output / name
        path.write_bytes(fetch_source(url))
        entries[name] = {"url": url, "sha256": sha256(path)}
    result = {"schema": 1, "status": "captured", "nanogpt_commit": commit, "files": entries,
              "mapping": "docs/scientific-runbook.md: source-equation mapping; project is an adaptation, not paper replication"}
    write_json(output / "source-evidence.json", result)
    return result
