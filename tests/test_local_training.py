import json
import hashlib
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from domain_shift_forgetting.local_corpus import (
    TokenStream, code_documents, json_write, load_corpus, sha256, wiki_articles, write_split,
)
from domain_shift_forgetting.local_training import settings, train


class FixtureEncoder:
    eot_token = 50256
    n_vocab = 50257

    def encode_ordinary(self, text):
        return list(text.encode())


def corpus(root: Path) -> None:
    root.mkdir()
    splits = {}
    for seed, key in enumerate(("web_train", "web_dev", "python_train", "python_dev")):
        values = np.random.default_rng(seed).integers(0, 50257, size=257, dtype=np.uint16)
        path = root / f"{key}.bin"
        values.tofile(path)
        splits[key] = {"tokens": len(values), "sha256": sha256(path)}
    json_write(root / "manifest.json", {"schema": 1, "tokenizer": "gpt2", "dtype": "<u2", "splits": splits})


def test_article_assembly_keeps_sections_inside_document():
    records = [{"text": s} for s in ["\n", " = A = \n", "first\n", " = = Section = = \n", "second\n", " = Bee = \n", "third\n"]]
    docs = list(wiki_articles(records))
    assert len(docs) == 2
    assert "Section" in docs[0]["text"] and "second" in docs[0]["text"]
    assert "third" not in docs[0]["text"]


def test_group_holdout_exact_dedup_and_token_cap(tmp_path):
    seen = set()
    dev = [{"group": "heldout", "source": "url1", "text": "dev text" * 160}]
    _, groups = write_split(iter(dev), tmp_path / "dev.bin", 1025, FixtureEncoder(), seen, set())
    training = [
        {"group": "heldout", "source": "url2", "text": "different function in heldout repo"},
        {"group": "other", "source": "url3", "text": dev[0]["text"]},
        {"group": "allowed", "source": "url4", "text": "new content " * 300},
    ]
    info, _ = write_split(iter(training), tmp_path / "train.bin", 1100, FixtureEncoder(), seen, groups)
    assert info["tokens"] == 1100
    assert info["heldout_group_rows_skipped"] == 1
    assert info["exact_duplicates_skipped"] == 1
    metadata = json.loads((tmp_path / "train.documents.jsonl").read_text())
    assert metadata["truncated"] and metadata["source"] == "url4"
    docs = list(code_documents([{"repository_name": "Owner/Repo", "whole_func_string": "code", "func_code_url": "url"}]))
    assert docs[0]["group"] == "owner/repo"


def test_mmap_windows_shift_targets_and_detect_corruption(tmp_path):
    root = tmp_path / "corpus"
    corpus(root)
    streams, _ = load_corpus(root, 16)
    stream = streams["web_train"]
    windows = stream.take(np.array([0, 1, stream.windows - 1]))
    assert windows.shape == (3, 17)
    assert windows[0, -1] == windows[1, 0]
    assert np.array_equal(windows[1], np.asarray(stream.tokens[16:33]))
    del stream, streams
    with (root / "python_train.bin").open("ab") as handle:
        handle.write(b"xx")
    with pytest.raises(ValueError, match="integrity"):
        load_corpus(root, 16)


def assert_state_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_state_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_state_equal(left, right)
    else:
        assert a == b


def test_resume_across_branch_transitions_matches_full_training(tmp_path):
    root = tmp_path / "corpus"
    corpus(root)
    config = settings(overrides=dict(context=8, batch_size=1, accumulation=1, precision="fp32",
                                    prefix_steps=2, continuation_steps=2, warmup=0, eval_every=2,
                                    eval_windows=1, checkpoint_every=1))
    device = torch.device("cpu")
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    train(config, root, full, device)
    train(config, root, resumed, device, max_updates=3)
    assert json.loads((resumed / "status.json").read_text())["status"] == "paused"
    train(config, root, resumed, device, resume=True, max_updates=2)
    train(config, root, resumed, device, resume=True)
    first = torch.load(full / "python-final.pt", map_location="cpu", weights_only=True)
    second = torch.load(resumed / "python-final.pt", map_location="cpu", weights_only=True)
    for key in ("model", "optimizer", "scaler", "sampler_rng", "torch_rng", "identity"):
        assert_state_equal(first[key], second[key])
    assert first["progress"]["endpoints"] == second["progress"]["endpoints"]
    history = second["progress"]["history"]
    starts = [row for row in history if row["stage"] in ("web", "python") and row["step"] == 0]
    assert starts[0]["web_dev_ce"] == starts[1]["web_dev_ce"]
    assert starts[0]["python_dev_ce"] == starts[1]["python_dev_ce"]
    assert json.loads((resumed / "summary.json").read_text())["status"] == "completed"
    with pytest.raises(ValueError, match="already complete"):
        train(config, root, resumed, device, resume=True)


def test_config_and_resume_identity_rejection(tmp_path):
    with pytest.raises(ValueError):
        settings(overrides={"context": 513})
    with pytest.raises(ValueError):
        settings(overrides={"accumulation": 0})
    root, run = tmp_path / "data", tmp_path / "run"
    corpus(root)
    run.mkdir()
    json_write(run / "manifest.json", {"identity": {}})
    with pytest.raises(ValueError, match="changed settings"):
        train(settings(overrides={"context": 8, "precision": "fp32"}), root, run, torch.device("cpu"), resume=True)


def test_download_recovers_incomplete_response_and_checks_hash(tmp_path, monkeypatch):
    import io
    from domain_shift_forgetting import local_corpus as module

    ranges = []

    class Response(io.BytesIO):
        def __init__(self, content, status, headers):
            super().__init__(content)
            self.status, self.headers = status, headers

    def open_response(request, timeout):
        ranges.append(request.get_header("Range"))
        if len(ranges) == 1:
            return Response(b"abc", 200, {"Content-Length": "6"})
        return Response(b"def", 206, {"Content-Length": "3", "Content-Range": "bytes 3-5/6"})

    monkeypatch.setattr(module, "urlopen", open_response)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    path = module.download("https://example.org/source", tmp_path / "source", hashlib.sha256(b"abcdef").hexdigest())
    assert path.read_bytes() == b"abcdef"
    assert ranges == [None, "bytes=3-"]
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.download("https://example.org/source", path, "0" * 64)


def test_source_catalog_excludes_upstream_test(monkeypatch):
    from domain_shift_forgetting import local_corpus as module
    monkeypatch.setattr(module, "remote_json", lambda url: [
        {"path": f"python/{split}-00000.parquet"} for split in ("train", "validation", "test")])
    names = [row["path"] for row in module.source_files(module.SOURCES["python"])]
    assert names == ["python/train-00000.parquet", "python/validation-00000.parquet"]


def test_parquet_preparation_roundtrip(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    tiktoken = pytest.importorskip("tiktoken")
    from domain_shift_forgetting import local_corpus as module
    fixtures = tmp_path / "fixtures"
    for domain in ("web", "python"):
        (fixtures / domain).mkdir(parents=True)
        for split in ("train", "validation"):
            if domain == "web":
                records = [{"text": f" = {split} article = \n"}, {"text": f"{split} prose " * 200}]
            else:
                records = [{"repository_name": f"owner/{split}", "whole_func_string": f"{split} code " * 200,
                            "func_code_url": f"https://example.org/{split}.py"}]
            pq.write_table(pa.Table.from_pylist(records), fixtures / domain / f"{split}-00000.parquet")

    def catalog(source):
        return [{"path": source["directory"] + f"/{split}-00000.parquet"} for split in ("train", "validation")]

    def local_download(url, path, expected_sha=None):
        if url.endswith("README.md"):
            path.write_text("Offline test fixture card")
            return path
        domain = "python" if "/python/" in url else "web"
        return fixtures / domain / url.rsplit("/", 1)[1]

    monkeypatch.setattr(module, "source_files", catalog)
    monkeypatch.setattr(module, "download", local_download)
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: FixtureEncoder())
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "tokenizer"))
    output = tmp_path / "prepared"
    module.prepare(output, tmp_path / "cache", 1025, 1025)
    streams, _ = load_corpus(output, 16)
    assert len(streams) == 4
    assert all(len(stream.tokens) == 1025 for stream in streams.values())
    with pytest.raises(FileExistsError):
        module.prepare(output, tmp_path / "cache", 1025, 1025)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA FP16")
def test_cuda_fp16_checkpoint_restores_scaler_optimizer_and_sampler(tmp_path):
    from domain_shift_forgetting.local_training import create_model, restore, save, update
    root = tmp_path / "data"
    corpus(root)
    streams, _ = load_corpus(root, 16)
    config = settings(overrides={"context": 16, "batch_size": 1, "accumulation": 1})
    device = torch.device("cuda")
    model, optimizer, scaler = create_model(config, device)
    assert all(group["fused"] for group in optimizer.param_groups)
    generator = torch.Generator().manual_seed(18)
    update(model, optimizer, scaler, streams["web_train"], config, generator, device)
    save(tmp_path / "state.pt", model, optimizer, scaler, generator, {"step": 1}, {"test": "fp16"})
    expected = update(model, optimizer, scaler, streams["web_train"], config, generator, device)
    expected_weights = {key: value.cpu().clone() for key, value in model.state_dict().items()}
    expected_scale = scaler.state_dict()
    restore(tmp_path / "state.pt", model, optimizer, scaler, generator, {"test": "fp16"})
    actual = update(model, optimizer, scaler, streams["web_train"], config, generator, device)
    assert actual == pytest.approx(expected, rel=1e-5, abs=1e-6)
    assert scaler.state_dict() == expected_scale
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value.cpu(), expected_weights[key], rtol=1e-5, atol=1e-6)
