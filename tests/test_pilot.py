"""Synthetic acceptance fixtures; never consume scientific corpora or seed budgets."""
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import zipfile

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from domain_shift_forgetting.analysis.decision import DecisionThresholds
from domain_shift_forgetting.local_corpus import sha256
from domain_shift_forgetting.pilot_control import (SessionClock, SessionLedger, digest_json, export_run, import_run,
                                                  load_policy, read_json, write_json)
from domain_shift_forgetting.pilot_data import aliases_from_row, filter_pool, materialize
from domain_shift_forgetting.pilot_report import event_path, report
from domain_shift_forgetting.protocol import FULL_PREFIX, FULL_CONTINUATION, QUICK_CONTINUATION, PREFIX_END, DIAGNOSTIC_CONTINUATION


@pytest.fixture
def archive_platform(tmp_path, monkeypatch):
    """Exercise the real Kaggle guard with a synthetic persisted-input mount."""
    from domain_shift_forgetting import pilot_control
    working = tmp_path / "platform-working"
    working.mkdir()
    monkeypatch.setattr(pilot_control, "KAGGLE_WORKING", working)
    monkeypatch.setattr(pilot_control, "KAGGLE_INPUT", tmp_path)


@pytest.fixture
def deterministic_cuda(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    previous = torch.are_deterministic_algorithms_enabled()
    threads = torch.get_num_threads()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)
        torch.set_num_threads(threads)


def test_policy_keeps_original_thresholds_and_effective_batch(tmp_path):
    config_path = Path(__file__).parents[1] / "configs/kaggle-stage1.json"
    config = load_policy(config_path)
    assert config["microbatch"] * config["accumulation"] * 512 == 16384
    assert config["decision_thresholds"] == asdict(DecisionThresholds())
    config["decision_thresholds"]["proceed_d"] = 0.02
    path = tmp_path / "changed.json"
    write_json(path, config)
    with pytest.raises(ValueError, match="threshold changes"):
        load_policy(path)


def test_live_session_deadline_does_not_restart_twelve_hours():
    clock = SessionClock.start(remaining_minutes=25, chunk_minutes=110, reserve_minutes=10, now=100)
    assert clock.deadline == 1600
    assert not clock.must_stop(now=999)
    assert clock.must_stop(now=1000)
    assert clock.must_stop(30, now=970)
    with pytest.raises(ValueError):
        SessionClock.start(5, 110, 10)
    with pytest.raises(ValueError):
        SessionClock.start(float("nan"), 110, 10)


def test_budget_reservation_survives_crash_and_rejects_decreasing_usage(tmp_path):
    path = tmp_path / "ledger.json"
    SessionLedger(path, reserved_seconds=600, external_gpu_hours=1)
    with pytest.raises(ValueError, match="Unclean"):
        SessionLedger(path, reserved_seconds=600, external_gpu_hours=1)
    resumed = SessionLedger(path, reserved_seconds=600, external_gpu_hours=1, recover_unclean=True)
    assert resumed.data["sessions"][0]["charged_seconds"] == 600
    resumed.finish(10, "paused")
    with pytest.raises(ValueError, match="cannot decrease"):
        SessionLedger(path, reserved_seconds=600, external_gpu_hours=0)


def test_verified_archive_keeps_generations_and_offloads_weight_history(tmp_path, archive_platform):
    root = tmp_path / "run"
    state = root / "runs/S101-RMS"
    state.mkdir(parents=True)
    generation = state / ".switch-example.pt"
    generation.write_bytes(b"trusted checkpoint fixture")
    write_json(state / "switch.pt.json", {"file": generation.name, "sha256": sha256(generation), "previous": None})
    (state / "switch.pt").write_bytes(generation.read_bytes())  # Alias must not be duplicated in ZIP.
    (state / "weights-prefix-763.pt").write_bytes(b"weight-only fixture")
    write_json(root / "freeze.json", {"fixture": True})
    archive = tmp_path / "saved.zip"
    export_run(root, archive)
    with zipfile.ZipFile(archive) as z:
        assert "runs/S101-RMS/switch.pt" not in z.namelist()
    restored = tmp_path / "restored"
    import_run(archive, restored)
    assert (restored / "runs/S101-RMS/.switch-example.pt").exists()
    assert not (restored / "runs/S101-RMS/weights-prefix-763.pt").exists()
    assert read_json(restored / "archived-weights.json")
    assert read_json(restored / "durable-prefixes.json")["prefixes"][str(Path("runs/S101-RMS"))] == sha256(generation)
    generation.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        export_run(root, tmp_path / "bad.zip")


def test_reports_archive_cannot_resume(tmp_path, archive_platform):
    root = tmp_path / "reports"
    root.mkdir()
    write_json(root / "status.json", {"status": "paused"})
    export_run(root, tmp_path / "reports.zip", reports_only=True)
    with pytest.raises(ValueError, match="reports-only"):
        import_run(tmp_path / "reports.zip", tmp_path / "bad-restore")


def test_original_alias_adapter_and_cross_split_dedup(tmp_path):
    pytest.importorskip("datasketch")
    assert aliases_from_row({"max_stars_repo_name": "Owner/Repo", "max_forks_repo_name": "owner/Other",
                             "repository_name": "https://github.com/owner/repo.git"}) == ("owner/other", "owner/repo")
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE docs(id TEXT PRIMARY KEY,domain TEXT,upstream TEXT,content_hash TEXT,tokens BLOB,metadata TEXT)")
    tokens = np.arange(100, dtype="<u2").tobytes()
    # Exact content duplicated across a held-out source and train must preserve held-out.
    for name, split in (("a", "train"), ("b", "validation")):
        db.execute("INSERT INTO docs VALUES (?,?,?,?,?,?)", (name, "web", split, "f" * 64, tokens, json.dumps({"aliases": []})))
    retained = filter_pool(db, tmp_path / "removals.jsonl")
    assert retained["web_train"] == []
    assert sum(len(v) for v in retained.values()) == 1
    db.close()


def test_materialization_owners_and_reserved_separation(tmp_path):
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE docs(id TEXT PRIMARY KEY,tokens BLOB,metadata TEXT)")
    retained, quotas = {}, {}
    for domain in ("web", "python"):
        for split in ("train", "dev", "test"):
            key = f"{domain}_{split}"
            retained[key], quotas[key] = [key], 513
            tokens = np.arange(512, dtype="<u2").tobytes()
            db.execute("INSERT INTO docs VALUES (?,?,?)", (key, tokens, json.dumps({"content_sha256": key, "aliases": []})))
    manifest = materialize(db, retained, tmp_path, {}, {}, quotas)
    assert set(manifest["splits"]) == {"web_train", "web_dev", "python_train", "python_dev"}
    assert (tmp_path / "reserved-test/web_test.bin").exists()
    assert not (tmp_path / "online/web_test.bin").exists()
    values = np.fromfile(tmp_path / "online/web_dev.bin", dtype="<u2")
    assert values[-1] == 50256 and len(values) == 513
    assert np.fromfile(tmp_path / "online/web_dev.owners.bin", dtype="<u4").tolist() == [0] * 513
    db.close()


def fake_event(frozen, seed, condition, stage, step, role, domain):
    count = 2097152 if role == "full" else 262144
    fraction = step / 6104 if stage != "prefix" else 0
    ce = 2.0 if domain == "web" else 3.0
    if stage == "python":
        ce += ((0.15 if condition == "Taper-minus" else 0.1) if domain == "web" else -0.5) * fraction
    stats = {"total": {"count": count, "ce_sum": ce * count}, "rare": {"count": 0, "ce_sum": 0.0},
             "documents": {"fixture": {"count": count, "ce_sum": ce * count}},
             "classes": {c: {"count": count if c == "A" else 0, "ce_sum": ce * count if c == "A" else 0.0} for c in "WAPX"}}
    return {"freeze_sha256": digest_json(frozen), "seed": seed, "condition": condition, "stage": stage, "step": step, "global_update": step if stage == "prefix" else PREFIX_END + step,
        "parent_sha256": None if stage == "prefix" else "b" * 64,
        "measurement": {"role": role, "domain": domain, "array_sha256": "a" * 64, "ce": ce,
                        "non_w_ce": ce, "ap_ce": ce, "statistics": stats}}


@pytest.mark.parametrize("paired_layout", [False, True])
def test_full_original_decision_from_bound_events_and_missingness(tmp_path, paired_layout):
    frozen = {"admitted": True, "policy": {"protocol": "test-fixture", "decision_thresholds": asdict(DecisionThresholds())},
              "arrays": {"splits": {f"{d}_dev": {"sha256": "a" * 64} for d in ("web", "python")}}}
    write_json(tmp_path / "freeze.json", frozen)
    if paired_layout:
        (tmp_path / "workers").mkdir()
    from domain_shift_forgetting.pilot_report import condition_root
    for seed in (101, 102, 103):
        for condition in ("RMS", "Taper-minus"):
            write_json(condition_root(tmp_path, condition) / "runs" / f"S{seed}-{condition}" / "switch.pt.json", {"sha256": "b" * 64})
            write_json(condition_root(tmp_path, condition) / "runs" / f"S{seed}-{condition}" / "completion.json", {"freeze_sha256": digest_json(frozen)})
            for stage, steps in (("prefix", (PREFIX_END,)), ("web", DIAGNOSTIC_CONTINUATION), ("python", DIAGNOSTIC_CONTINUATION)):
                for step in steps:
                    write_json(condition_root(tmp_path, condition) / "diagnostics" / f"S{seed}-{condition}" / f"{stage}-{step}.json", {"freeze_sha256": digest_json(frozen)})
            for stage in ("prefix", "web", "python"):
                for role in ("full", "quick"):
                    steps = (FULL_PREFIX if role == "full" else (PREFIX_END,)) if stage == "prefix" else (FULL_CONTINUATION if role == "full" else QUICK_CONTINUATION)
                    for step in steps:
                        for domain in ("web", "python"):
                            write_json(event_path(tmp_path, seed, condition, stage, step, role, domain),
                                       fake_event(frozen, seed, condition, stage, step, role, domain))
    result = report(tmp_path)
    assert result["decision"]["category"] == "PROCEED TO DESIGN THE NEXT STUDY"
    assert result["decision"]["statistics"]["mean_D"] == pytest.approx(0.05)
    terminal = event_path(tmp_path, 103, "RMS", "python", 6104, "full", "web")
    corrupted = read_json(terminal)
    corrupted["parent_sha256"] = "c" * 64
    write_json(terminal, corrupted)
    with pytest.raises(ValueError, match="frozen prefix"):
        report(tmp_path)
    terminal.unlink()
    assert report(tmp_path)["decision"]["category"] == "INVALID OR INCOMPLETE"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for actual FP16 update replay")
def test_original_ordered_update_fp16_resume_and_activation(tmp_path, deterministic_cuda):
    from domain_shift_forgetting.local_corpus import TokenStream
    from domain_shift_forgetting.local_training import save, restore
    from domain_shift_forgetting.pilot_runner import make_state, ordered_update
    from domain_shift_forgetting.kaggle import snapshot, compare
    from domain_shift_forgetting.models.norms import TaperNorm
    path = tmp_path / "fixture.bin"
    np.random.default_rng(7).integers(0, 50257, 512 * 64 + 1, dtype=np.uint16).tofile(path)
    stream = TokenStream(path, 512)
    policy = {"microbatch": 2, "accumulation": 16, "precision": "fp16"}
    device = torch.device("cuda")
    model, optimizer, scaler, sampler = make_state(997, "Taper-minus", policy, device)
    # Synthetic near-boundary EMA fixture; not a trained scientific state.
    model.completed_updates.fill_(762)
    for norm in model.modules():
        if isinstance(norm, TaperNorm):
            norm.ema_updates.fill_(762)
    ordered_update(model, optimizer, scaler, stream, np.arange(32), policy, device)
    assert all(bool(m.calibrated) for m in model.modules() if isinstance(m, TaperNorm))
    save(tmp_path / "state.pt", model, optimizer, scaler, sampler, {"cursor": 32}, {"fixture": True})
    ordered_update(model, optimizer, scaler, stream, np.arange(32, 64), policy, device)
    expected = snapshot({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()})
    for norm in model.modules():
        if isinstance(norm, TaperNorm):
            assert int(optimizer.state[norm.gamma_tilde]["step"]) == 1
    state = restore(tmp_path / "state.pt", model, optimizer, scaler, sampler, {"fixture": True})
    assert state["cursor"] == 32
    ordered_update(model, optimizer, scaler, stream, np.arange(32, 64), policy, device)
    compare({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()}, expected)


def test_kaggle_auto_extracted_archive_restore_and_tampering(tmp_path, archive_platform):
    from domain_shift_forgetting.pilot_control import import_directory
    root = tmp_path / "run"
    root.mkdir()
    write_json(root / "freeze.json", {"fixture": True})
    export_run(root, tmp_path / "archive.zip")
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(tmp_path / "archive.zip") as archive:
        archive.extractall(extracted)
    import_directory(extracted, tmp_path / "restored")
    assert read_json(tmp_path / "restored/freeze.json") == {"fixture": True}
    write_json(extracted / "freeze.json", {"fixture": False})
    with pytest.raises(ValueError, match="hash mismatch"):
        import_directory(extracted, tmp_path / "tampered")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA diagnostic probe fixture")
def test_diagnostic_hooks_are_forward_only_and_removed(deterministic_cuda):
    from types import SimpleNamespace
    from domain_shift_forgetting.models.transformer import Transformer
    from domain_shift_forgetting.protocol import Condition
    from domain_shift_forgetting.pilot_diagnostics import diagnostics
    from domain_shift_forgetting.kaggle import compare
    # Short synthetic sequences exercise every hook without scientific exposure.
    class Stream:
        windows = 256
        def take(self, indices):
            return np.tile(np.arange(33, dtype=np.int64), (len(indices), 1))
    model = Transformer(Condition.TAPER_MINUS).cuda()
    before = {name: value.detach().cpu().clone() for name, value in model.named_buffers()}
    arrays = SimpleNamespace(streams={"web_train": Stream(), "python_train": Stream()}, classes=np.full(50257, "A"))
    result = diagnostics(model, arrays, np.zeros(50257, dtype=bool), microbatch=8, precision="fp16")
    compare(dict(model.named_buffers()), before)
    assert result["gradient_tokens"] == 0
    assert len(result["sites"]) == 24
    assert result["sites"]["0.attention.z"]["all"]["web"]["positions"] == 256 * 32
    assert result["sites"]["0.attention.z"]["position_mask"]["web"]["positions"] == 256 * 16
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("kind", ["zip", "directory"])
def test_kaggle_restore_rejects_ephemeral_source(tmp_path, monkeypatch, kind):
    from domain_shift_forgetting import pilot_control
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.setattr(pilot_control, "KAGGLE_WORKING", working)
    monkeypatch.setattr(pilot_control, "KAGGLE_INPUT", tmp_path / "input")
    destination = tmp_path / "restored"
    operation = pilot_control.import_run if kind == "zip" else pilot_control.import_directory
    with pytest.raises(ValueError, match="persisted archive"):
        operation(working / "ephemeral", destination)
    assert not destination.exists()


def test_free_budget_amendment_preserves_science_and_requires_explicit_record(tmp_path):
    configs = Path(__file__).parents[1] / "configs"
    original = load_policy(configs / "kaggle-stage1-24h-baseline.json")
    amended = load_policy(configs / "kaggle-stage1.json")
    allowed = {"protocol", "optimizer_hours_cap", "active_gpu_hours_cap", "budget_amendment"}
    assert {k: v for k, v in original.items() if k not in allowed} == {k: v for k, v in amended.items() if k not in allowed}
    assert amended["active_gpu_hours_cap"] == amended["optimizer_hours_cap"] == 50
    changed = tmp_path / "policy.json"
    del amended["budget_amendment"]
    write_json(changed, amended)
    with pytest.raises(ValueError, match="documented budget"):
        load_policy(changed)


def test_free_budget_ledger_continues_across_weekly_reset(tmp_path):
    path = tmp_path / "ledger.json"
    # Previous quota use remains cumulative even though the platform refreshed.
    first = SessionLedger(path, reserved_seconds=600, external_gpu_hours=30, cap_hours=50)
    first.finish(5, "paused")
    resumed = SessionLedger(path, reserved_seconds=600, external_gpu_hours=30, cap_hours=50)
    assert len(resumed.data["sessions"]) == 2
    resumed.finish(5, "paused")
    with pytest.raises(ValueError, match="cannot decrease"):
        SessionLedger(path, reserved_seconds=600, external_gpu_hours=0, cap_hours=50)
    with pytest.raises(ValueError, match="exhausted"):
        SessionLedger(path, reserved_seconds=600, external_gpu_hours=50, cap_hours=50)
