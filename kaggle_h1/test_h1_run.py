"""Equivalence and plumbing tests for kaggle_h1/h1_run.py against the tested src/ package.

Run from the repository root:  .venv/Scripts/python -m pytest kaggle_h1/test_h1_run.py -q
Real-data tests use data/kaggle-online (the prepared Kaggle corpus) when present.
"""
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "kaggle-online"
sys.path.insert(0, str(ROOT / "src"))

from domain_shift_forgetting import protocol as P  # noqa: E402
from domain_shift_forgetting.models import transformer as src_model  # noqa: E402
from domain_shift_forgetting.analysis import decision as src_decision  # noqa: E402
from domain_shift_forgetting.analysis import matching as src_matching  # noqa: E402
from domain_shift_forgetting.analysis import contrasts as src_contrasts  # noqa: E402
from domain_shift_forgetting.analysis import tails as src_tails  # noqa: E402


def load_h1():
    spec = importlib.util.spec_from_file_location("h1_run_under_test", ROOT / "kaggle_h1" / "h1_run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


H1 = load_h1()
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
needs_data = pytest.mark.skipif(not (DATA / "online" / "manifest.json").exists(), reason="prepared corpus absent")


def test_protocol_schedule_and_clocks_match_src():
    assert H1.FULL_PREFIX == P.FULL_PREFIX
    assert H1.QUICK_CONTINUATION == P.QUICK_CONTINUATION
    assert H1.FULL_CONTINUATION == P.FULL_CONTINUATION
    assert H1.DIAGNOSTIC_CONTINUATION == P.DIAGNOSTIC_CONTINUATION
    assert (H1.CALIBRATION_END, H1.GATE_END, H1.PREFIX_END, H1.CONTINUATION_UPDATES, H1.TRAJECTORY_END) == \
        (P.CALIBRATION_END, P.GATE_END, P.PREFIX_END, P.CONTINUATION_UPDATES, P.TRAJECTORY_END)
    for u in range(1, P.TRAJECTORY_END + 1):
        assert H1.learning_rate(u) == P.learning_rate(u)
        assert H1.gate(u) == P.gate(u)
    assert len(H1.QUICK_CONTINUATION) == 80 and len(H1.FULL_CONTINUATION) == 25
    # Planned token totals (protocol page 01)
    assert (H1.PREFIX_END + 2 * H1.CONTINUATION_UPDATES) * H1.TOKENS_PER_UPDATE * 6 == 2_100_166_656


@pytest.mark.parametrize("condition", ["RMS", "Taper-minus"])
def test_paired_initialization_matches_src_make_state(condition):
    model, _, _ = H1.make_state(101, condition, torch.device("cpu"))
    random.seed(101); np.random.seed(101); torch.manual_seed(101)
    canonical = src_model.Transformer(P.Condition.RMS)
    reference = canonical if condition == "RMS" else src_model.Transformer(P.Condition(condition))
    if reference is not canonical:
        src_model.copy_canonical_initialization(canonical, reference)
    mine, theirs = model.state_dict(), reference.state_dict()
    assert mine.keys() == theirs.keys()
    for key in mine:
        assert torch.equal(mine[key], theirs[key]), key


def test_rms_and_taper_share_identical_initial_tensors():
    rms, _, _ = H1.make_state(102, "RMS", torch.device("cpu"))
    taper, _, _ = H1.make_state(102, "Taper-minus", torch.device("cpu"))
    shared = dict(rms.named_parameters())
    for name, parameter in taper.named_parameters():
        if name.endswith("gamma_tilde"):
            continue
        assert torch.equal(parameter, shared[name]), name


def _src_twin(model, condition):
    twin = src_model.Transformer(P.Condition(condition))
    twin.load_state_dict(model.state_dict())
    return twin


@pytest.mark.parametrize("condition", ["RMS", "Taper-minus"])
def test_training_updates_match_src_effective_update_cpu(condition):
    """Same data, FP32 CPU: h1 ordered_update == src effective_update over calibration."""
    from domain_shift_forgetting.training.step import effective_update, make_optimizer
    torch.manual_seed(0)
    model, optimizer, scaler = H1.make_state(103, condition, torch.device("cpu"))
    twin = _src_twin(model, condition)
    twin_optimizer = make_optimizer(twin)

    class Stream:
        def __init__(self):
            rng = np.random.default_rng(1)
            self.tokens = rng.integers(0, 50257, size=40 * 512 + 1)

        def take(self, indices):
            offsets = np.asarray(indices)[:, None] * 512 + np.arange(513)
            return self.tokens[offsets].astype(np.int64)

    stream = Stream()
    for step in range(3):
        indices = np.arange(32) + step
        metrics = H1.ordered_update(model, optimizer, scaler, stream, indices, 8, torch.device("cpu"))
        rows = torch.from_numpy(stream.take(indices))
        result = effective_update(twin, twin_optimizer, [(rows[i:i + 8, :-1], rows[i:i + 8, 1:]) for i in range(0, 32, 8)],
                                  bf16=False)
        assert math.isclose(metrics["ce"], result.ce, rel_tol=1e-5)
        for (name, a), b in zip(model.state_dict().items(), twin.state_dict().values()):
            torch.testing.assert_close(a.float(), b.float(), rtol=2e-5, atol=2e-6, msg=name)


def test_taper_forward_matches_src_across_gate_states():
    torch.manual_seed(5)
    model, _, _ = H1.make_state(101, "Taper-minus", torch.device("cpu"))
    for module in model.modules():
        if isinstance(module, H1.TaperNorm):
            module.calibrated.fill_(True)
            module.c.fill_(0.37)
            module.gamma_tilde.data.uniform_(0.5, 1.5)
    twin = _src_twin(model, "Taper-minus")
    tokens = torch.randint(0, 50257, (2, 64))
    for update in (1, 763, 764, 2000, 6103, 6104, 9000):
        model.completed_updates.fill_(update)
        twin.completed_updates.fill_(update)
        a, _ = model(tokens, update=update)
        b, _ = twin(tokens, update=update)
        assert torch.equal(a, b), update


def test_analysis_rules_match_src_on_random_evidence():
    rng = random.Random(7)
    for trial in range(3000):
        def seed_row(seed):
            quick = {u: rng.uniform(-0.1, 0.1) for u in P.QUICK_CONTINUATION}
            matched = {u: (None if rng.random() < 0.2 else rng.uniform(-0.05, 0.05)) for u in (1525, 3050, 6104)}
            values = dict(seed=seed, d=rng.choice([rng.uniform(-0.06, 0.08), 0.015, 0.03, -0.03]),
                          d3050=rng.uniform(-0.03, 0.05), q=rng.uniform(-0.05, 0.08),
                          absolute_relative_prefix_gap=rng.choice([0.0, 0.01, 0.019, 0.021, 0.05]),
                          rms_non_w_improvement=rng.choice([0.04, 0.05, 0.3]),
                          taper_non_w_improvement=rng.choice([0.04, 0.05, 0.3]), quick_d=quick,
                          matched_differences=matched)
            return values
        rows = [seed_row(s) for s in (101, 102, 103)]
        mine = H1.classify([H1.SeedEvidence(**r) for r in rows], primary_complete=True,
                           correctness_and_data_passed=True)
        theirs = src_decision.classify([src_decision.SeedEvidence(**r) for r in rows], primary_complete=True,
                                       correctness_and_data_passed=True)
        assert mine.category == theirs.category, (trial, mine, theirs)
        assert mine.common_target_update == theirs.common_target_update
        assert dict(mine.statistics) == dict(theirs.statistics)
    incomplete = H1.classify([], primary_complete=False, correctness_and_data_passed=True)
    assert incomplete.category == "INVALID OR INCOMPLETE"


def test_matching_and_contrasts_match_src():
    rng = random.Random(11)
    for _ in range(2000):
        updates = sorted({0} | set(rng.sample(range(1, 6105), 30)) | {1525, 3050, 6104})
        def series():
            level = rng.uniform(1.5, 2.5)
            return [(u, max(0.0, level - rng.uniform(0, 1.0) * (u / 6104) + rng.uniform(-0.05, 0.05)),
                     rng.uniform(3.0, 3.5)) for u in updates]
        rms, taper = series(), series()
        for target in (1525, 3050, 6104):
            a = H1.matched_forgetting([H1.AdaptationPoint(*p) for p in rms], [H1.AdaptationPoint(*p) for p in taper], target)
            b = src_matching.matched_forgetting([src_matching.AdaptationPoint(*p) for p in rms],
                                                [src_matching.AdaptationPoint(*p) for p in taper], target)
            assert (a[1] is None) == (b[1] is None)
            if a[1] is not None:
                assert a[1] == b[1] and a[0].update == b[0].update
        values = [rng.uniform(2.5, 4.0) for _ in range(6)]
        mine = H1.contrast(H1.BranchCE(*values))
        theirs = src_contrasts.contrast(src_contrasts.BranchCE(*values))
        assert mine.d == theirs.d and mine.q == theirs.q and mine.g_web == theirs.g_web
    points = [(u, 3.0 - u * 1e-6) for u in P.FULL_PREFIX]
    assert H1.prefix_slope(points) == src_contrasts.prefix_slope(points)
    docs = [(f"d{i}", rng.randint(1, 900), rng.uniform(-0.2, 0.3)) for i in range(500)]
    d = math.fsum(c * x for _, c, x in docs) / sum(c for _, c, _ in docs)
    mine = H1.summarize_documents([H1.DocumentEffect(*r) for r in docs], d)
    theirs = src_tails.summarize_documents([src_tails.DocumentEffect(*r) for r in docs], d)
    assert (mine.top_signed_sum, mine.trimmed_token_weighted_mean, mine.unweighted_median,
            mine.outlier_concentrated) == (theirs.top_signed_sum, theirs.trimmed_token_weighted_mean,
                                           theirs.unweighted_median, theirs.outlier_concentrated)


class _ShimStream:
    def __init__(self, tokens):
        self.tokens = tokens
        self.windows = (len(tokens) - 1) // 512

    def take(self, indices):
        offsets = indices[:, None] * 512 + np.arange(513)
        return np.asarray(self.tokens[offsets], dtype=np.int64)


@needs_cuda
@needs_data
@pytest.mark.parametrize("domain", ["web", "python"])
def test_vectorized_evaluation_matches_src_evaluator(domain):
    from domain_shift_forgetting.pilot_arrays import evaluate as src_evaluate
    data = H1.Data(DATA / "online", DATA / "orders")
    rare = data.seed_orders(101)["rare"]
    device = torch.device("cuda")
    model, _, _ = H1.make_state(101, "Taper-minus", device)

    class Arrays:
        manifest = data.manifest
        classes = data.class_letters
        key = f"{domain}_dev"
        dev = data.dev[domain]
        streams = {key: _ShimStream(dev.tokens)}
        owners = {key: np.fromfile(DATA / "online" / data.manifest["splits"][key]["owners"], dtype="<u4")}
        docs = {key: [{"id": i} for i in dev.doc_ids]}

    mine = H1.evaluate(model, data, domain, "quick", rare, 8, device)
    theirs = src_evaluate(model, Arrays, domain, role="quick", rare=rare, microbatch=8, precision="fp16")
    stats = theirs["statistics"]
    assert mine["total"][1] == stats["total"]["count"] == 262144
    assert math.isclose(mine["ce"], theirs["ce"], rel_tol=1e-9)
    assert math.isclose(mine["non_w_ce"], theirs["non_w_ce"], rel_tol=1e-9)
    for letter in "WAPX":
        assert mine["classes"][letter][1] == stats["classes"][letter]["count"]
        assert math.isclose(mine["classes"][letter][0], stats["classes"][letter]["ce_sum"], rel_tol=1e-9, abs_tol=1e-6)
    assert mine["rare"][1] == stats["rare"]["count"]
    full = H1.evaluate(model, data, domain, "full", rare, 16, device)
    assert sum(full["doc_counts"]) == 2_097_152
    sums = {doc: v["ce_sum"] for doc, v in stats["documents"].items()}
    # quick-set documents must match the full-set prefix of the same documents where wholly contained
    assert abs(math.fsum(full["doc_sums"]) - full["total"][0]) / 2_097_152 < 1e-9
    assert len(sums) > 10


@needs_cuda
def test_checkpoint_resume_reproduces_uninterrupted_updates(tmp_path):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda")

    class Stream:
        def __init__(self):
            self.tokens = np.random.default_rng(3).integers(0, 50257, size=200 * 512 + 1)

        def take(self, indices):
            offsets = np.asarray(indices)[:, None] * 512 + np.arange(513)
            return self.tokens[offsets].astype(np.int64)

    stream = Stream()
    identity = {"config_sha256": "x", "seed": 101, "condition": "Taper-minus"}
    H1.CALIBRATION_END, saved = 3, H1.CALIBRATION_END  # cross the calibration freeze within the test
    try:
        a = H1.make_state(101, "Taper-minus", device)
        for step in range(6):
            H1.ordered_update(*a, stream, np.arange(32) + 32 * step, 8, device)
        b = H1.make_state(101, "Taper-minus", device)
        for step in range(3):
            H1.ordered_update(*b, stream, np.arange(32) + 32 * step, 8, device)
        H1.save_checkpoint(tmp_path / "latest.pt", *b, {"stage": "prefix", "step": 3}, identity)
        c = H1.make_state(101, "Taper-minus", device)
        progress = H1.load_checkpoint(tmp_path / "latest.pt", *c, identity)
        assert progress["step"] == 3
        for step in range(3, 6):
            H1.ordered_update(*c, stream, np.arange(32) + 32 * step, 8, device)
        # GPU embedding/attention backward kernels are nondeterministic: compare the resumed run's
        # deviation with a second uninterrupted run's deviation (protocol: no bitwise claim).
        a2 = H1.make_state(101, "Taper-minus", device)
        for step in range(6):
            H1.ordered_update(*a2, stream, np.arange(32) + 32 * step, 8, device)
        rerun = resumed = 0.0
        for (name, x), y, z in zip(a[0].state_dict().items(), a2[0].state_dict().values(), c[0].state_dict().values()):
            if x.dtype.is_floating_point:
                rerun = max(rerun, (x - y).abs().max().item())
                resumed = max(resumed, (x - z).abs().max().item())
            else:
                assert torch.equal(x, z), name
        assert resumed <= 3 * rerun + 1e-5, (resumed, rerun)
        assert a[2].get_scale() == c[2].get_scale()
        assert bool(c[0].blocks[0].attention_norm.calibrated)
    finally:
        H1.CALIBRATION_END = saved
        torch.use_deterministic_algorithms(False)


def _run_main(work, inputs, hours, extra_env=None):
    env = dict(os.environ, H1_SESSION_HOURS=str(hours), PYTHONIOENCODING="utf-8", **(extra_env or {}))
    command = [sys.executable, str(ROOT / "kaggle_h1" / "h1_run.py"), "main", "--mini", "--inputs", str(inputs),
               "--work", str(work), "--online", str(DATA / "online"), "--orders", str(DATA / "orders"),
               "--microbatch", "8", "--eval-microbatch", "8"]
    result = subprocess.run(command, env=env, capture_output=True, text=True, encoding="utf-8", timeout=3600)
    print(result.stdout[-6000:])
    print(result.stderr[-3000:])
    assert result.returncode == 0, result.stderr[-3000:]
    return result.stdout


@needs_cuda
@needs_data
def test_mini_protocol_end_to_end_with_session_interruptions(tmp_path):
    """Plumbing only: tiny schedule, two workers sharing one GPU, three sessions, final report."""
    empty = tmp_path / "empty-inputs"
    empty.mkdir()
    s1, s2, s3 = tmp_path / "s1", tmp_path / "s2", tmp_path / "s3"
    two = {"H1_TEST_TWO_WORKERS_ONE_GPU": "1"}
    out = _run_main(s1, empty, 0.012, {"H1_ALLOW_FRESH_START": "1", **two})  # ~43 s session: interrupted
    assert "NEW experiment created" in out
    first = json.loads((s1 / "h1state" / "decision-report.json").read_text())
    assert first["decision"]["category"] == "INVALID OR INCOMPLETE"
    # A missing previous state with fresh start disabled must refuse to train.
    env = dict(os.environ, H1_SESSION_HOURS="0.01", PYTHONIOENCODING="utf-8")
    refused = subprocess.run([sys.executable, str(ROOT / "kaggle_h1" / "h1_run.py"), "main", "--mini", "--inputs",
                              str(empty), "--work", str(tmp_path / "refuse"), "--online", str(DATA / "online"),
                              "--orders", str(DATA / "orders")], env=env, capture_output=True, text=True,
                             encoding="utf-8")
    assert refused.returncode == 4
    _run_main(s2, s1, 0.02, two)            # resume from s1's output, interrupted again
    out = _run_main(s3, s2, 0.5, two)       # resume from s2's output and finish
    report = json.loads((s3 / "h1state" / "decision-report.json").read_text())
    assert report["complete"], report["missing_events_first_50"]
    assert report["decision"]["category"] != "INVALID OR INCOMPLETE"
    assert len(report["seeds"]) == 3
    sessions = (s3 / "h1state" / "sessions.jsonl").read_text().strip().splitlines()
    assert len(sessions) == 3
    for seed in (101, 102, 103):
        for condition in ("RMS", "Taper-minus"):
            run = s3 / "h1state" / "runs" / f"S{seed}-{condition}"
            assert (run / "completion.json").exists() and (run / "switch.pt").exists()
            assert not (run / "latest.pt").exists()
    # A completed experiment is not trained again.
    again = _run_main(tmp_path / "s4", s3, 0.5, two)
    assert "Nothing to train" in again
