"""Concurrency, joint recovery and budget tests using synthetic state only."""
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from domain_shift_forgetting.paired_control import (supervise, exclusive_run, worker_environment,
                                                    propagate_archive_indexes, CONDITIONS)
from domain_shift_forgetting.pilot_control import read_json, write_json, export_run, import_run, import_directory, load_policy
from domain_shift_forgetting.pilot_report import condition_root, event_path
from domain_shift_forgetting.paired_preflight import projected_work


def job(script, status=None):
    result = {"command": [sys.executable, "-c", script], "env": os.environ.copy()}
    if status is not None:
        result["status"] = status
    return result


def test_worker_mapping_and_scientific_policy():
    root = Path(__file__).parents[1]
    paired = load_policy(root / "configs/kaggle-stage1-paired.json")
    single = load_policy(root / "configs/kaggle-stage1.json")
    for key in ("seeds", "conditions", "microbatch", "accumulation", "precision", "decision_thresholds"):
        assert paired[key] == single[key]
    assert paired["execution"]["notebook_hours_cap"] == 50
    assert worker_environment("0")["CUDA_VISIBLE_DEVICES"] == "0"
    assert worker_environment("1")["CUDA_VISIBLE_DEVICES"] == "1"


def test_os_lock_excludes_second_coordinator_and_releases(tmp_path):
    with exclusive_run(tmp_path):
        with pytest.raises(RuntimeError, match="already"):
            with exclusive_run(tmp_path):
                pass
    with exclusive_run(tmp_path):
        pass


def test_worker_failure_stops_and_joins_peer(tmp_path):
    stop = tmp_path / "STOP"
    marker = tmp_path / "peer-exited"
    peer = f"from pathlib import Path; import time\np=Path({str(stop)!r})\nwhile not p.exists(): time.sleep(.02)\nPath({str(marker)!r}).write_text('stopped')"
    with pytest.raises(RuntimeError, match="Worker failed"):
        supervise({"bad": job("raise SystemExit(3)"), "peer": job(peer)}, tmp_path / "logs", stop,
                  deadline=time.monotonic() + 10, grace_seconds=5)
    assert marker.read_text() == "stopped"
    assert not (tmp_path / "logs/writers-active.json").exists()
    assert set(read_json(tmp_path / "logs/worker-lifetimes.json")) == {"bad", "peer"}


def test_prefix_pause_stops_peer_and_success_does_not(tmp_path):
    stop = tmp_path / "STOP"
    status = tmp_path / "status.json"
    write_json(status, {"status": "awaiting_durable_prefix"})
    marker = tmp_path / "joined"
    peer = f"from pathlib import Path; import time\nwhile not Path({str(stop)!r}).exists(): time.sleep(.02)\nPath({str(marker)!r}).write_text('done')"
    supervise({"prefix": job("pass", status), "peer": job(peer, status)}, tmp_path / "logs", stop,
              deadline=time.monotonic() + 10, stop_on_pause=True, grace_seconds=5)
    assert marker.exists()
    # A fully completed worker must not prematurely stop its still-active peer.
    stop2 = tmp_path / "STOP2"
    write_json(status, {"status": "completed"})
    peer2 = f"from pathlib import Path; import time\ntime.sleep(.3)\nassert not Path({str(stop2)!r}).exists()"
    supervise({"done": job("pass", status), "peer": job(peer2, status)}, tmp_path / "logs2", stop2,
              deadline=time.monotonic() + 10, stop_on_pause=True, grace_seconds=5)


def test_deadline_kills_unresponsive_worker_before_export(tmp_path):
    with pytest.raises(RuntimeError, match="Worker failed"):
        supervise({"hung": job("import time; time.sleep(30)")}, tmp_path / "logs", tmp_path / "STOP",
                  deadline=time.monotonic() + .1, grace_seconds=.2)


def test_projection_has_separate_worker_costs():
    evaluation = {"full": 0, "quick": 0, "diagnostics": 0, "checkpoint": 0}
    times = {"calibration": 1, "intermediate": 1, "zero": 1}
    rms = projected_work("RMS", times, evaluation)
    taper = projected_work("Taper-minus", times, evaluation)
    assert rms["optimizer_hours"] == taper["optimizer_hours"] == 3 * (9156 + 2 * 6104) / 3600


@pytest.mark.parametrize("extracted", [False, True])
def test_joint_archive_preserves_two_proofs_and_worker_prefix_indexes(tmp_path, monkeypatch, extracted):
    from domain_shift_forgetting import pilot_control
    from domain_shift_forgetting.local_corpus import sha256
    monkeypatch.setattr(pilot_control, "KAGGLE_WORKING", tmp_path)
    monkeypatch.setattr(pilot_control, "KAGGLE_INPUT", tmp_path)
    root = tmp_path / "root"
    for c in CONDITIONS:
        worker = root / "workers" / c
        state = worker / "runs" / f"S101-{c}"
        state.mkdir(parents=True)
        gen = state / ".switch-fixture.pt"
        gen.write_bytes(c.encode())
        write_json(state / "switch.pt.json", {"file": gen.name, "sha256": sha256(gen), "previous": None})
        (worker / "resume-expected.pt").write_bytes(b"required replay proof")
        (state / "weights-prefix-763.pt").write_bytes(b"historical weights")
    archive = tmp_path / "paired.zip"
    export_run(root, archive)
    restored = tmp_path / "restored"
    if extracted:
        import zipfile
        source = tmp_path / "mounted"
        with zipfile.ZipFile(archive) as z:
            z.extractall(source)
        import_directory(source, restored)
    else:
        import_run(archive, restored)
    propagate_archive_indexes(restored)
    for c in CONDITIONS:
        child = restored / "workers" / c
        assert (child / "resume-expected.pt").read_bytes() == b"required replay proof"
        assert str(Path(f"runs/S101-{c}")) in read_json(child / "durable-prefixes.json")["prefixes"]
        assert f"runs/S101-{c}/weights-prefix-763.pt" in read_json(child / "archived-weights.json")
        assert condition_root(restored, c) == child
        assert event_path(restored, 101, c, "web", 0, "full", "web").is_relative_to(child)


def test_isolated_condition_runner_survives_joint_export_and_partial_resume(tmp_path, monkeypatch):
    """Exercise the real stage controller with tiny CPU optimizer/checkpoint state."""
    from domain_shift_forgetting import pilot_runner as runner, pilot_control
    from domain_shift_forgetting.local_corpus import sha256
    from domain_shift_forgetting.local_training import restore
    monkeypatch.setattr(pilot_control, "KAGGLE_WORKING", tmp_path)
    monkeypatch.setattr(pilot_control, "KAGGLE_INPUT", tmp_path)
    monkeypatch.setattr(runner, "PREFIX_END", 1)
    monkeypatch.setattr(runner, "CONTINUATION_UPDATES", 1)
    monkeypatch.setattr(runner, "FULL_PREFIX", (1,))
    for name in ("FULL_CONTINUATION", "QUICK_CONTINUATION", "DIAGNOSTIC_CONTINUATION"):
        monkeypatch.setattr(runner, name, (0, 1))
    monkeypatch.setattr(runner, "source_identity", lambda: {})
    monkeypatch.setattr(runner, "runtime_identity", lambda device: {})
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(runner, "PilotArrays", lambda path: SimpleNamespace(identity="fixture", streams={"web_train": 0, "python_train": 100}))
    monkeypatch.setattr(runner, "load_orders", lambda *args: {"web": np.arange(64), "python": np.arange(32), "rare": []})
    monkeypatch.setattr(runner, "evaluate", lambda *args, **kwargs: {"synthetic": True})
    monkeypatch.setattr(runner, "diagnostics", lambda *args, **kwargs: {"synthetic": True})
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.register_buffer("completed_updates", torch.tensor(0))
    def make(*args):
        model = Model()
        return model, torch.optim.AdamW(model.parameters(), lr=.01), torch.amp.GradScaler("cpu", enabled=False), torch.Generator().manual_seed(7)
    observed = []
    def update(model, optimizer, scaler, stream, indices, *args):
        observed.append((stream, int(indices[0]), int(model.completed_updates)))
        optimizer.zero_grad()
        (model.weight * (1 + stream)).backward()
        optimizer.step()
        model.completed_updates.add_(1)
        return {"ce": 1, "clipped": False, "overflow_retries": 0}
    monkeypatch.setattr(runner, "make_state", make)
    monkeypatch.setattr(runner, "ordered_update", update)
    orders = tmp_path / "orders"
    write_json(orders / "manifest.json", {})
    root = tmp_path / "run"
    policy = load_policy(Path(__file__).parents[1] / "configs/kaggle-stage1-paired.json")
    frozen = {"layout": "paired-v1", "admitted": True, "code": {}, "runtimes": {c: {} for c in CONDITIONS},
              "policy": policy, "arrays_sha256": "fixture", "orders_sha256": sha256(orders / "manifest.json"),
              "benchmark": {"prior_gpu_hours": 0}, "run_order": [[997, c] for c in CONDITIONS]}
    write_json(root / "freeze.json", frozen)
    for c in CONDITIONS:
        child = root / "workers" / c
        write_json(child / "freeze.json", frozen)
        runner.run(child, tmp_path, orders, remaining_minutes=120, external_gpu_hours=0, worker_condition=c)
        assert read_json(child / "status.json")["status"] == "awaiting_durable_prefix"
        assert not (child / "runs" / f"S997-{CONDITIONS[1 - CONDITIONS.index(c)]}").exists()
    export_run(root, tmp_path / "joint.zip")
    recovered = tmp_path / "recovered"
    import_run(tmp_path / "joint.zip", recovered)
    propagate_archive_indexes(recovered)
    for c in CONDITIONS:
        child = recovered / "workers" / c
        runner.run(child, tmp_path, orders, remaining_minutes=120, external_gpu_hours=0, worker_condition=c, max_updates=1)
        assert read_json(child / "status.json")["status"] == "paused"
        runner.run(child, tmp_path, orders, remaining_minutes=120, external_gpu_hours=0, worker_condition=c)
        assert read_json(child / "status.json")["status"] == "completed"
    assert observed.count((0, 0, 0)) == 2  # two independent prefixes
    assert observed.count((0, 32, 1)) == 2  # web child continues its order
    assert observed.count((100, 0, 1)) == 2  # Python child restores same prefix clock


def test_full_export_requires_joined_writers_and_closed_ledger(tmp_path):
    from domain_shift_forgetting.paired_control import assert_quiescent
    marker = tmp_path / "coordinator/attempt/writers-active.json"
    write_json(marker, {"fixture": True})
    with pytest.raises(ValueError, match="writers"):
        assert_quiescent(tmp_path)
    marker.unlink()
    write_json(tmp_path / "notebook-ledger.json", {"sessions": [{"status": "running"}]})
    with pytest.raises(ValueError, match="ledger"):
        assert_quiescent(tmp_path)
    write_json(tmp_path / "notebook-ledger.json", {"sessions": [{"status": "paused"}]})
    assert_quiescent(tmp_path)


def test_preflight_export_excludes_synthetic_test_weights(tmp_path):
    import zipfile
    root = tmp_path / "preflight"
    test_folder = root / "workers/RMS/test-tmp/fixture"
    test_folder.mkdir(parents=True)
    (test_folder / "weights-prefix-1.pt").write_bytes(b"synthetic fixture")
    archive = tmp_path / "preflight.zip"
    export_run(root, archive)
    with zipfile.ZipFile(archive) as z:
        assert all("test-tmp" not in name for name in z.namelist())
