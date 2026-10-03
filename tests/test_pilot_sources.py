import io
import json
import sys
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from domain_shift_forgetting import pilot_sources as sources


def test_cards_use_secret_and_dataset_download(tmp_path, monkeypatch):
    card = tmp_path / "README.md"
    card.write_bytes(b"approved card")
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(card)
    monkeypatch.setenv("HF_TOKEN", "fixture-secret")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    assert sources.fetch_source("https://huggingface.co/datasets/bigcode/the-stack-dedup/raw/main/README.md") == b"approved card"
    assert calls == [dict(repo_id="bigcode/the-stack-dedup", repo_type="dataset", revision="main", filename="README.md", token="fixture-secret")]


def test_public_sources_never_receive_hf_secret(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fixture-secret")
    def open_public(request, timeout):
        assert not request.has_header("Authorization")
        assert "fixture-secret" not in str(request.headers)
        return io.BytesIO(b"public source")
    monkeypatch.setattr(sources, "urlopen", open_public)
    assert sources.fetch_source("https://arxiv.org/html/2602.10408v1") == b"public source"


def test_failed_source_identified_without_sensitive_error(monkeypatch):
    def fail(*args, **kwargs):
        raise HTTPError("https://example.com/?token=secret", 401, "secret", {}, None)
    monkeypatch.setattr(sources, "urlopen", fail)
    with pytest.raises(RuntimeError) as caught:
        sources.fetch_source("https://arxiv.org/html/2602.10408v1")
    assert "arxiv.org/html/2602.10408v1" in str(caught.value)
    assert "401" in str(caught.value)
    assert "secret" not in str(caught.value)


def test_capture_retries_partial_output_without_false_success(tmp_path, monkeypatch):
    failed = [False]
    def fetch(url):
        if "commits/master" in url:
            return json.dumps({"sha": "a" * 40}).encode()
        if "the-stack-dedup" in url and not failed[0]:
            failed[0] = True
            raise RuntimeError("HTTP 401")
        return url.encode()
    monkeypatch.setattr(sources, "fetch_source", fetch)
    out = tmp_path / "upstream"
    with pytest.raises(RuntimeError):
        sources.capture(out)
    assert not (out / "source-evidence.json").exists()
    assert sources.capture(out)["status"] == "captured"
    with pytest.raises(FileExistsError):
        sources.capture(out)
