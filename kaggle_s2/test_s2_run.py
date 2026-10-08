"""Equivalence, contract and plumbing tests for the Study 2 code (kaggle_s2/). Light CPU work only.

Run from the repository root:  .venv/Scripts/python -m pytest kaggle_s2/test_s2_run.py -q
Nothing here uses a GPU or trains a model. Checks that read the full local pilot corpus (data/kaggle-online/) are
opt-in (S2_HEAVY_TESTS=1). GPU-side checks (lineage of the real switch states, the mini protocol end to end,
throughput) run in the Kaggle smoke notebook (build_notebooks.py smoke).
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # never touch the local GPU

from collections import Counter  # noqa: E402
from dataclasses import asdict  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import sqlite3  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "kaggle_s2"
PILOT_STATE = ROOT / "reports" / "h1-kaggle" / "h1state"
DATA = ROOT / "data" / "kaggle-online"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


H1 = load("h1_run_reference", ROOT / "kaggle_h1" / "h1_run.py")
S2 = load("s2_run", HERE / "s2_run.py")
PD = load("prepare_domain", HERE / "prepare_domain.py")
PROBE = load("probe_domains", HERE / "probe_domains.py")
BUILD = load("s2_build_notebooks", HERE / "build_notebooks.py")
S2M = load("s2_run_mini", HERE / "s2_run.py")
S2M.use_mini_protocol()
H1M = load("h1_run_mini", ROOT / "kaggle_h1" / "h1_run.py")
H1M.use_mini_protocol()
CPU = torch.device("cpu")
heavy = pytest.mark.skipif(os.environ.get("S2_HEAVY_TESTS") != "1", reason="opt-in: S2_HEAVY_TESTS=1")
needs_data = pytest.mark.skipif(not (DATA / "online" / "manifest.json").exists(), reason="pilot corpus absent")
CLASSES = np.array(list("WAPX") * (50257 // 4) + ["X"])


# =============================================================================
# 1. The runner is the pilot's code
# =============================================================================
VERBATIM = ("learning_rate", "gate", "stage_limit", "use_mini_protocol", "digest_json", "sha256_file", "read_json",
            "write_json", "log", "TrainStream", "Condition", "RMSNorm", "TaperNorm", "Attention", "Block",
            "Transformer", "copy_canonical_initialization", "make_state", "ordered_update", "sensitivity_mask",
            "EventLog", "save_weights", "LossSum", "EvaluationSums", "SeedUncertainty", "prefix_slope",
            "AdaptationPoint", "Match", "first_crossing", "matched_forgetting", "DocumentEffect", "ClassEffect",
            "_four_way", "decompose", "TailSummary", "summarize_documents", "SeedEvidence", "Decision",
            "stats_from_record", "gpu_count")
SCHEDULE = ("CONTEXT", "VOCAB", "EOS", "SEQUENCES_PER_UPDATE", "TOKENS_PER_UPDATE", "WARMUP_END", "CALIBRATION_END",
            "GATE_END", "PREFIX_END", "CONTINUATION_UPDATES", "TRAJECTORY_END", "FULL_PREFIX", "QUICK_CONTINUATION",
            "FULL_CONTINUATION", "DIAGNOSTIC_CONTINUATION", "PERSISTENCE_POINTS", "D_MID_POINT",
            "TRANSIENT_MAX_UPDATE", "MATCH_MAX_BRACKET", "FULL_WINDOWS", "QUICK_WINDOWS", "DIAG_WINDOWS",
            "WORKER_RESERVE_SECONDS", "EVAL_RESERVE_SECONDS", "DECISION_THRESHOLDS", "CLASS_CODES", "CONDITIONS")


@pytest.mark.parametrize("name", VERBATIM)
def test_copied_pilot_code_is_verbatim(name):
    assert inspect.getsource(getattr(S2, name)) == inspect.getsource(getattr(H1, name))


def test_protocol_constants_equal_the_pilots_and_mini_schedules_align():
    for name in SCHEDULE:
        assert getattr(S2, name) == getattr(H1, name), name
        assert getattr(S2M, name) == getattr(H1M, name), name  # a mini "pilot" switch fits the mini Study 2
    assert S2.PILOT_SEEDS == H1.SEEDS and S2.SEEDS == S2.PILOT_SEEDS + S2.FRESH_SEEDS
    assert S2M.MINI and not S2.MINI
    for name in ("EOS", "CONTEXT", "VOCAB", "PREFIX_END", "CONTINUATION_UPDATES", "SEQUENCES_PER_UPDATE",
                 "PILOT_SEEDS", "FRESH_SEEDS", "SEEDS", "S2_ARRAYS_SCHEMA", "S2_ORDERS_SCHEMA"):
        assert getattr(PD, name) == getattr(S2, name), name
    assert PD.PILOT_ONLINE_MANIFEST_SHA256 == S2.PILOT_PINS["online_manifest_sha256"]
    assert PROBE.SELECTION_SCHEMA == PD.SELECTION_SCHEMA and PROBE.SITES == S2.SITES and len(S2.SITES) == 12


def test_schedule_is_the_pilots_and_the_x_branch_mirrors_the_python_branch():
    for stage in ("prefix", "web", "python"):
        for step in range(S2.stage_limit(stage) + 1):
            assert S2.events_at(stage, step) == H1.events_at(stage, step), (stage, step)
    for step in range(S2.CONTINUATION_UPDATES + 1):
        evals, diag = H1.events_at("python", step)
        assert S2.events_at("x", step) == ([(r, "x" if d == "python" else d) for r, d in evals], diag)
    # The data each update reads: the pilot's run_job offsets, and the X order from its start.
    assert S2.stage_data("prefix", 5) == ("web", 160)
    assert S2.stage_data("web", 5) == ("web", (S2.PREFIX_END + 5) * 32)
    assert S2.stage_data("python", 5) == ("python", 160) and S2.stage_data("x", 5) == ("x", 160)
    assert S2.job_stages(101) == ("x",) and S2.job_stages(104) == ("prefix", "web", "python", "x")
    # Planned supervised tokens (protocol Sec. 3): 3 X branches + 3 full runs, per condition.
    assert sum(S2.run_updates(s) for s in S2.SEEDS) * S2.TOKENS_PER_UPDATE * 2 == 3_300_261_888


@pytest.mark.parametrize("seed", [101, 104])
@pytest.mark.parametrize("condition", ["RMS", "Taper-minus"])
def test_paired_initialization_equals_the_pilots(seed, condition):
    mine, theirs = S2.make_state(seed, condition, CPU)[0].state_dict(), H1.make_state(seed, condition, CPU)[0].state_dict()
    assert mine.keys() == theirs.keys() and all(torch.equal(mine[k], theirs[k]) for k in mine)


def test_pilot_switch_checkpoints_restore_into_the_study2_runner(tmp_path):
    model, optimizer, scaler = H1.make_state(101, "Taper-minus", CPU)
    model.completed_updates.fill_(S2.PREFIX_END)
    pilot_identity = {"config_sha256": S2.PILOT_PINS["config_sha256"], "seed": 101, "condition": "Taper-minus"}
    H1.save_checkpoint(tmp_path / "switch.pt", model, optimizer, scaler, {"stage": "prefix", "step": S2.PREFIX_END},
                       pilot_identity)
    mine = S2.make_state(101, "Taper-minus", CPU)
    progress = S2.load_checkpoint(tmp_path / "switch.pt", *mine, pilot_identity, schema="h1-ckpt-1")
    assert progress["step"] == S2.PREFIX_END and int(mine[0].completed_updates) == S2.PREFIX_END
    assert all(torch.equal(v, mine[0].state_dict()[k]) for k, v in model.state_dict().items())
    with pytest.raises(ValueError):  # a pilot state is never read as a Study 2 checkpoint, or for another run
        S2.load_checkpoint(tmp_path / "switch.pt", *S2.make_state(101, "Taper-minus", CPU), pilot_identity)
    with pytest.raises(ValueError):
        S2.load_checkpoint(tmp_path / "switch.pt", *S2.make_state(101, "Taper-minus", CPU),
                           pilot_identity | {"seed": 102}, schema="h1-ckpt-1")
    own = {"config_sha256": "study2", "seed": 101, "condition": "Taper-minus"}
    S2.save_checkpoint(tmp_path / "latest.pt", *mine, {"stage": "x", "step": 0}, own)
    again = S2.make_state(101, "Taper-minus", CPU)
    assert S2.load_checkpoint(tmp_path / "latest.pt", *again, own) == {"stage": "x", "step": 0}
    corrupted = bytearray((tmp_path / "latest.pt").read_bytes())
    corrupted[len(corrupted) // 2] ^= 0xFF
    (tmp_path / "latest.pt").write_bytes(bytes(corrupted))
    with pytest.raises(ValueError, match="checksum"):
        S2.load_checkpoint(tmp_path / "latest.pt", *again, own)


def _dev_set(tmp_path: Path, rng, windows: int) -> dict:
    tokens = rng.integers(0, 50257, size=windows * 512 + 1).astype("<u2")
    sizes = [400, 500, 600]
    owners = np.repeat(np.arange(4), sizes + [len(tokens) - sum(sizes)]).astype("<u4")
    tokens.tofile(tmp_path / "dev.bin")
    owners.tofile(tmp_path / "dev.owners.bin")
    (tmp_path / "dev.documents.json").write_text(json.dumps({"documents": [{"id": f"d{i}"} for i in range(4)]}))
    return {"file": "dev.bin", "owners": "dev.owners.bin", "documents": "dev.documents.json", "sha256": "synthetic"}


def test_evaluation_equals_the_pilots(tmp_path, monkeypatch):
    for module in (S2, H1):
        monkeypatch.setattr(module, "FULL_WINDOWS", 3)
        monkeypatch.setattr(module, "QUICK_WINDOWS", 2)
    rng = np.random.default_rng(0)
    row = _dev_set(tmp_path, rng, 3)
    codes = np.array([S2.CLASS_CODES[c] for c in CLASSES])
    rare = rng.random(50257) < 0.3

    class Mine:
        dev = {"web": S2.DevSet(tmp_path, row, "web")}
        class_codes = codes
        dev_sha256 = {"web": "synthetic"}

    class Theirs:
        dev = {"web": H1.DevSet(tmp_path, {"splits": {"web_dev": row}}, "web")}
        class_codes = codes
        manifest = {"splits": {"web_dev": row}}

    model = S2.make_state(101, "Taper-minus", CPU)[0]
    for role in ("quick", "full"):
        assert S2.evaluate(model, Mine, "web", role, rare, 2, CPU) == H1.evaluate(model, Theirs, "web", role, rare, 2, CPU)


def test_diagnostics_equal_the_pilots(tmp_path, monkeypatch):
    for module in (S2, H1):
        monkeypatch.setattr(module, "DIAG_WINDOWS", 3)
    rng = np.random.default_rng(1)
    streams = {}
    for name in ("web", "python"):
        rng.integers(0, 50257, size=5 * 512 + 1).astype("<u2").tofile(tmp_path / f"{name}.bin")
        streams[name] = S2.TrainStream(tmp_path / f"{name}.bin")
    rare = rng.random(50257) < 0.3

    class Theirs:
        train = streams
        class_letters = CLASSES

    model = S2.make_state(102, "Taper-minus", CPU)[0]
    theirs = H1.diagnostics(model, Theirs, rare, 2, CPU)
    mine = S2.diagnostics(model, streams, CLASSES, rare, 2, CPU)
    for site, views in theirs["sites"].items():
        for view, row in views.items():
            for domain in ("web", "python"):
                assert mine["sites"][site][view][domain] == row[domain]
            assert mine["sites"][site][view]["log_k_vs_web"]["python"] == row["log_k"]
            assert mine["sites"][site][view]["near_zero_energy"] == row["near_zero_energy"]
    for key in ("branch_residual_rms_ratios", "attention_entropy_nats_per_head", "gains", "calibration_c",
                "embedding_norms"):
        assert mine[key] == theirs[key], key
    assert mine["forward_labels_per_domain"] == {"web": 3 * 512, "python": 3 * 512}


def _random_evidence(rng: random.Random, seed: int) -> dict:
    return dict(seed=seed, d=rng.choice([rng.uniform(-0.06, 0.08), 0.015, 0.03, -0.03]),
                d3050=rng.uniform(-0.03, 0.05), q=rng.uniform(-0.05, 0.08),
                absolute_relative_prefix_gap=rng.choice([0.0, 0.01, 0.019, 0.021, 0.05]),
                rms_non_w_improvement=rng.choice([0.04, 0.05, 0.3]), taper_non_w_improvement=rng.choice([0.04, 0.05, 0.3]),
                quick_d={u: rng.uniform(-0.1, 0.1) for u in H1.QUICK_CONTINUATION},
                matched_differences={u: (None if rng.random() < 0.2 else rng.uniform(-0.05, 0.05))
                                     for u in H1.PERSISTENCE_POINTS})


def test_decision_rules_equal_the_pilots_and_generalize_to_six_seeds():
    rng = random.Random(7)
    for trial in range(3000):
        rows = [_random_evidence(rng, s) for s in H1.SEEDS]
        mine = S2.classify([S2.SeedEvidence(**r) for r in rows], expected_seeds=H1.SEEDS, primary_complete=True,
                           correctness_and_data_passed=True)
        theirs = H1.classify([H1.SeedEvidence(**r) for r in rows], primary_complete=True,
                             correctness_and_data_passed=True)
        assert (mine.category, mine.common_target_update, dict(mine.statistics)) == \
            (theirs.category, theirs.common_target_update, dict(theirs.statistics)), trial
    good = [S2.SeedEvidence(**(_random_evidence(rng, s) | dict(
        d=0.05, d3050=0.03, q=0.05, absolute_relative_prefix_gap=0.003, rms_non_w_improvement=1.0,
        taper_non_w_improvement=1.0, matched_differences={u: 0.01 for u in S2.PERSISTENCE_POINTS}))) for s in S2.SEEDS]
    assert S2.classify(good, expected_seeds=S2.SEEDS, primary_complete=True,
                       correctness_and_data_passed=True).category == "PROCEED TO DESIGN THE NEXT STUDY"
    assert S2.classify(good[:5], expected_seeds=S2.SEEDS, primary_complete=True,
                       correctness_and_data_passed=True).category == "INVALID OR INCOMPLETE"
    assert S2.classify(good, expected_seeds=S2.SEEDS, primary_complete=True,
                       correctness_and_data_passed=False).category == "INVALID OR INCOMPLETE"
    one_negative = good[:5] + [S2.SeedEvidence(**(good[5].__dict__ | {"d": -0.01}))]
    assert S2.classify(one_negative, expected_seeds=S2.SEEDS, primary_complete=True,
                       correctness_and_data_passed=True).category == "INCONCLUSIVE"


def test_contrasts_and_intervals_equal_the_pilots():
    rng = random.Random(11)
    for _ in range(2000):
        values = [rng.uniform(2.5, 4.0) for _ in range(6)]
        mine, theirs = S2.contrast(S2.BranchCE(*values)), H1.contrast(H1.BranchCE(*values))
        assert (mine.d, mine.g_web, mine.q, mine.f_rms_shift, mine.f_taper_shift) == \
            (theirs.d, theirs.g_web, theirs.q, theirs.f_rms_python, theirs.f_taper_python)
        seeds = [rng.uniform(-0.05, 0.05) for _ in range(3)]
        assert asdict(S2.seed_uncertainty(seeds, 4.303)) == asdict(H1.seed_uncertainty(seeds))
    summary = S2.interval_summary([0.001, -0.002, 0.0, 0.003, -0.001, 0.002])
    assert summary["t95_multiplier"] == 2.571 and summary["t90_multiplier"] == 2.015
    assert summary["equivalent_within_bound"] and not summary["t95_excludes_zero"]
    assert not S2.interval_summary([0.02, 0.025, 0.03])["equivalent_within_bound"]


def test_lineage_comparison_gates_restored_switch_states():
    reference = {("full", "web"): {"ce": 4.5}, ("quick", "web"): {"ce": 4.6}}
    same = S2.lineage_comparison(dict(reference), reference, None, None)
    assert same["passed"] and same["max_ce_diff"] == 0.0 and not same["diag_compared"]
    drifted = {k: {"ce": v["ce"] + 2e-3} for k, v in reference.items()}
    assert not S2.lineage_comparison(drifted, reference, None, None)["passed"]
    assert not S2.lineage_comparison({}, reference, None, None)["passed"]  # nothing compared is not a pass
    diag = {"sites": {"0.attention.h": {"all": {"web": {"mean_squared_norm": 1.0}, "python": {"mean_squared_norm": 2.0}}}}}
    moved = {"sites": {"0.attention.h": {"all": {"web": {"mean_squared_norm": 1.01}, "python": {"mean_squared_norm": 2.0}}}}}
    assert not S2.lineage_comparison(dict(reference), reference, moved, diag)["passed"]


# =============================================================================
# 2. The pilot assets Study 2 reuses
# =============================================================================
def test_pilot_pins_are_the_repositorys_pilot_records():
    config = json.loads((PILOT_STATE / "config.json").read_text(encoding="utf-8"))
    body = {k: v for k, v in config.items() if k not in ("config_sha256", "checkpoint_seconds")}
    assert config["config_sha256"] == S2.PILOT_PINS["config_sha256"] == S2.digest_json(body)
    assert config["data"]["online_manifest_sha256"] == S2.PILOT_PINS["online_manifest_sha256"]
    assert config["data"]["orders_manifest_sha256"] == S2.PILOT_PINS["orders_manifest_sha256"]
    corpus_manifest = ROOT / "reports" / "h1-kaggle" / "corpus" / "manifest.json"
    assert hashlib.sha256(corpus_manifest.read_bytes()).hexdigest() == S2.PILOT_PINS["online_manifest_sha256"]
    assert json.loads((PILOT_STATE / "H1-STATE.json").read_text())["experiment_id"] == S2.PILOT_PINS["experiment_id"]
    session = json.loads((PILOT_STATE / "sessions.jsonl").read_text().splitlines()[0])
    runner = hashlib.sha256((ROOT / "kaggle_h1" / "h1_run.py").read_bytes()).hexdigest()
    assert session["code_sha256"] == runner == S2.PILOT_PINS["runner_sha256"]
    assert S2.PilotReference(PILOT_STATE).verify(hash_switch_files=False) == S2.PILOT_PINS


def test_selection_statistic_and_manipulation_check_reproduce_finding_a(tmp_path):
    pilot = S2.PilotReference(PILOT_STATE)
    models = [(s, c) for s in S2.PILOT_SEEDS for c in S2.CONDITIONS]
    energies = {key: PROBE.recorded_energies(pilot, *key) for key in models}
    stats = PROBE.gap_statistic(energies, "python", models)
    assert (f"{stats['per_condition']['RMS']:.3f}", f"{stats['per_condition']['Taper-minus']:.3f}") == ("0.097", "0.123")
    assert stats["S"] == pytest.approx(sum(stats["per_condition"].values()) / 2)
    measures = S2.scale_measures(S2.Records(tmp_path, pilot), S2.PILOT_SEEDS, "python")
    paper = {"RMS": (0.097, 0.206, 0.207, -0.206, -0.196, 0.042, 36, 30),
             "Taper-minus": (0.123, 0.054, 0.077, 0.040, 0.060, 0.042, 6, 3)}  # paper Table 5 (Finding A)
    for condition, expected in paper.items():
        row = measures[condition]
        got = (row["switch_gap_abs"], row["web_branch_change_abs"], row["shift_branch_change_abs"],
               row["web_branch_change_signed"], row["shift_branch_change_signed"], row["specific_change_abs"])
        assert tuple(round(v, 3) for v in got) == expected[:6], condition
        assert (row["web_branch_shrank"], row["shift_branch_shrank"], row["pairs"]) == (*expected[6:], 36)


def test_selection_rule_picks_the_largest_gap_with_a_fixed_tie_order():
    order = PD.CANDIDATE_ORDER
    assert PROBE.select({"mc4-de": 0.1, "mc4-ru": 0.3, "mc4-zh": 0.2, "openwebmath": 0.05}) == "mc4-ru"
    assert PROBE.select({"mc4-de": 0.3, "mc4-ru": 0.3, "mc4-zh": 0.3, "openwebmath": 0.3}) == order[0]
    assert PROBE.select({"mc4-de": 0.1, "mc4-ru": 0.1, "mc4-zh": 0.1, "openwebmath": 0.2}) == "openwebmath"
    near_tie = {"mc4-de": 0.3, "mc4-ru": 0.3 + 1e-13, "mc4-zh": 0.1, "openwebmath": 0.2}
    assert PROBE.ranking(near_tie) == ["mc4-de", "mc4-ru", "openwebmath", "mc4-zh"]  # rank 0 is always select()
    assert PROBE.ranking(near_tie)[0] == PROBE.select(near_tie)


# =============================================================================
# 3. The Study 2 report on synthetic records with known answers (mini protocol)
# =============================================================================
def _record(m, stage, step, role, domain, ce, parent, docs=8):
    n = (m.FULL_WINDOWS if role == "full" else m.QUICK_WINDOWS) * m.CONTEXT
    counts = {"W": n // 8, "A": n // 2, "P": n // 4}
    counts["X"] = n - sum(counts.values())
    sums = {k: ce * c for k, c in counts.items()}
    total = math.fsum(sums.values())
    other = n - counts["W"]
    record = {"domain": domain, "role": role, "array_sha256": "synthetic", "total": [total, n],
              "classes": {k: [sums[k], counts[k]] for k in "WAPX"}, "rare": [ce * 1024, 1024], "ce": total / n,
              "non_w_ce": sum(sums[k] for k in "APX") / other, "ap_ce": ce, "stage": stage, "step": step,
              "global_update": step if stage == "prefix" else m.PREFIX_END + step,
              "parent_sha256": None if stage == "prefix" else parent, "seconds": 0.0}
    if role == "full":
        record["doc_sums"], record["doc_counts"] = [total / docs] * docs, [n // docs] * docs
    return record


def _diag(m, stage, step, domain_key, energies):
    return {"stage": stage, "step": step, "role": "diag", "domain": domain_key,
            "global_update": step if stage == "prefix" else m.PREFIX_END + step, "status": "ok", "seconds": 0.0,
            "measurement": {"sites": {f"{site}.h": {"all": {d: {"mean_squared_norm": e} for d, e in energies.items()}}
                                      for site in S2.SITES}}}


def _write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def build_synthetic_study(tmp: Path, delta_x: dict, delta_py: dict):
    """A complete mini Study 2: a fake pilot (seeds 101-103) and Study 2's twelve runs with planted effects.

    Web CE: prefix 4.5 (+0.005 for Taper-minus); web branch -0.1 over the branch; Python/X branches +1.6, plus
    delta * s / C for Taper-minus only. The shifted domain's CE falls 8 -> 5 identically in both conditions, so
    every matched target is an exact match and D = D(C) = delta, Q = D, D(mid) = delta / 2."""
    m = S2M
    C, P = m.CONTINUATION_UPDATES, m.PREFIX_END
    pilot_dir, state = tmp / "pilot" / "h1state", tmp / "work" / "s2state"

    def web_ce(seed, condition, stage, step):
        base = 4.5 + (0.005 if condition == "Taper-minus" else 0.0)
        if stage == "prefix":
            return base + 1.5 * (P - step) / P
        if stage == "web":
            return base - 0.1 * step / C
        delta = (delta_x if stage == "x" else delta_py)[seed] if condition == "Taper-minus" else 0.0
        return base + 1.6 * step / C + delta * step / C

    def other_ce(stage, step):
        return 8.0 - 3.0 * step / C if stage in ("python", "x") else 8.0 + 0.01 * step

    def energies(stage, step, domains):
        web = {"prefix": 1.0, "web": 1.0 - 0.19 * step / C, "python": 1.0 - 0.19 * step / C,
               "x": 1.0 - 0.4 * step / C}[stage]
        return {d: {"web": web, "python": 0.8, "x": 0.5}[d] for d in domains}

    def stage_rows(seed, condition, stage, parent, diag_key, diag_domains):
        rows = []
        full = m.FULL_PREFIX if stage == "prefix" else m.FULL_CONTINUATION
        quick = (P,) if stage == "prefix" else m.QUICK_CONTINUATION
        for role, steps in (("full", full), ("quick", quick)):
            for step in steps:
                for domain in m.STAGE_DOMAINS[stage]:
                    ce = web_ce(seed, condition, stage, step) if domain == "web" else other_ce(stage, step)
                    rows.append(_record(m, stage, step, role, domain, ce, parent))
        diag_steps = (P,) if stage == "prefix" else m.DIAGNOSTIC_CONTINUATION
        rows += [_diag(m, stage, step, diag_key, energies(stage, step, diag_domains)) for step in diag_steps]
        return rows

    published = []
    for seed in m.PILOT_SEEDS:
        for condition in m.CONDITIONS:
            run = pilot_dir / "runs" / f"S{seed}-{condition}"
            switch = f"pilot-switch-{seed}-{condition}"
            rows = sum((stage_rows(seed, condition, s, switch, "both", ("web", "python"))
                        for s in ("prefix", "web", "python")), [])
            _write(run / "events.jsonl", rows)
            m.write_json(run / "switch.pt.json", {"sha256": switch})
            m.write_json(run / "completion.json", {"switch_sha256": switch})
        ce = {(c, b): web_ce(seed, c, b, C) for c in m.CONDITIONS for b in ("web", "python")}
        prefix = {c: web_ce(seed, c, "prefix", P) for c in m.CONDITIONS}
        published.append({"seed": seed, "endpoint": {"d": m.contrast(m.BranchCE(
            prefix["RMS"], prefix["Taper-minus"], ce["RMS", "web"], ce["RMS", "python"], ce["Taper-minus", "web"],
            ce["Taper-minus", "python"])).d}})
    m.write_json(pilot_dir / "config.json", {"config_sha256": "fake-pilot-config"})
    m.write_json(pilot_dir / "H1-STATE.json", {"experiment_id": "fake-pilot"})
    m.write_json(pilot_dir / "decision-report.json", {"seeds": published})
    pilot = m.PilotReference(pilot_dir)
    domain = {"id": "mc4-zh", "label": "Chinese web text (mC4 zh)", "repo": "allenai/c4", "revision": "r",
              "selection_sha256": "s"}
    cfg = m.config_document(8, 4, 16, pilot.pins, domain) | {"data": {}, "config_sha256": "fake-s2-config",
                                                               "checkpoint_seconds": 900}
    m.write_json(state / "config.json", cfg)
    for seed in m.SEEDS:
        for condition in m.CONDITIONS:
            run = state / "runs" / f"S{seed}-{condition}"
            if seed in m.PILOT_SEEDS:
                switch, rows = f"pilot-switch-{seed}-{condition}", []
            else:
                switch = f"s2-switch-{seed}-{condition}"
                m.write_json(run / "switch.pt.json", {"sha256": switch})
                rows = stage_rows(seed, condition, "prefix", switch, "all", m.DIAG_DOMAINS)
            for stage in m.job_stages(seed):
                if stage == "prefix":
                    continue
                rows += stage_rows(seed, condition, stage, switch, "all", m.DIAG_DOMAINS)
                rows.append({"stage": stage, "step": 0, "role": "lineage", "domain": "switch", "max_ce_diff": 0.0,
                             "max_energy_rel_diff": 0.0, "passed": True})
            _write(run / "events.jsonl", rows)
            m.write_json(run / "completion.json", {"seed": seed, "condition": condition})
    return state, pilot, [f"doc{i}" for i in range(8)]


def test_report_recovers_planted_effects_and_applies_the_registered_rules(tmp_path):
    delta_x = {s: 0.05 + 0.001 * i for i, s in enumerate(S2.SEEDS)}
    delta_py = {s: -0.002 + 0.0005 * i for i, s in enumerate(S2.SEEDS)}
    state, pilot, docs = build_synthetic_study(tmp_path, delta_x, delta_py)
    report = S2M.build_report(state, pilot, docs)
    assert report["problems"] == [] and report["failures"] == [] and report["missing_event_count"] == 0
    assert report["complete"] and report["h2"]["complete"] and report["r2"]["complete"]
    assert report["h2"]["decision"]["category"] == "PROCEED TO DESIGN THE NEXT STUDY"
    for row in report["h2"]["seeds"]:
        assert row["evidence"]["d"] == pytest.approx(delta_x[row["seed"]], abs=1e-12)
        assert row["evidence"]["d3050"] == pytest.approx(delta_x[row["seed"]] / 2, abs=1e-12)
        assert row["evidence"]["q"] == pytest.approx(delta_x[row["seed"]], abs=1e-12)
        assert row["evidence"]["matched_differences"][S2M.CONTINUATION_UPDATES] == pytest.approx(delta_x[row["seed"]])
        assert row["adaptation_non_w_improvement"]["RMS"] == pytest.approx(3.0)
        assert row["tail"]["d"] == pytest.approx(delta_x[row["seed"]], abs=1e-9)
    assert report["h2"]["uncertainty"]["mean"] == pytest.approx(np.mean(list(delta_x.values())))
    assert report["h2"]["fresh_seed_sensitivity"]["n"] == 3
    assert report["r2"]["decision"]["category"] == "STOP — SMALL OBSERVED EFFECT"
    assert report["r2"]["uncertainty"]["equivalent_within_bound"]
    assert [row["seed"] for row in report["r2"]["seeds"]] == list(S2.FRESH_SEEDS)
    assert report["pooled_python_descriptive"]["n"] == 6
    reading = report["manipulation_check"]["reading"]
    assert reading == {"switch_gap_larger_for_x": True, "specific_change_larger_for_x": True}
    assert report["manipulation_check"]["x"]["RMS"]["switch_gap_abs"] == pytest.approx(abs(0.5 * math.log(0.5)))
    assert report["document_bootstrap"]["documents"] == 8
    assert len(report["lineage"]) == 3 * 2 + 3 * 2 * 3 and all(v["passed"] for v in report["lineage"].values())
    assert report["summary_lines"][0].startswith("H2 ANSWER (web -> mc4-zh): YES")


def test_report_reads_equivalence_and_never_turns_gaps_into_findings(tmp_path):
    delta_x = {s: -0.002 + 0.0007 * i for i, s in enumerate(S2.SEEDS)}
    state, pilot, docs = build_synthetic_study(tmp_path / "ok", delta_x, delta_x)
    report = S2M.build_report(state, pilot, docs)
    assert report["h2"]["decision"]["category"] == "STOP — SMALL OBSERVED EFFECT"
    assert report["h2"]["uncertainty"]["equivalent_within_bound"]
    # A missing X-branch record: H2 is incomplete, R2 (fresh web/Python branches) is still decided.
    state, pilot, docs = build_synthetic_study(tmp_path / "missing", delta_x, delta_x)
    log = state / "runs" / "S105-RMS" / "events.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    dropped = [r for r in rows if (r["stage"], r["step"], r["role"], r["domain"])
               == ("x", S2M.CONTINUATION_UPDATES, "full", "web")]
    assert len(dropped) == 1
    _write(log, [r for r in rows if r is not dropped[0]])
    report = S2M.build_report(state, pilot, docs)
    assert report["h2"]["decision"]["category"] == "INVALID OR INCOMPLETE" and not report["h2"]["complete"]
    assert report["r2"]["decision"]["category"] == "STOP — SMALL OBSERVED EFFECT"
    # A failed lineage check, a pilot record that does not descend from its switch, or a numerical failure all
    # invalidate both decisions.
    for case in ("lineage", "parent", "numerical"):
        state, pilot, docs = build_synthetic_study(tmp_path / case, delta_x, delta_x)
        if case == "lineage":
            log = state / "runs" / "S102-Taper-minus" / "events.jsonl"
            log.write_text(log.read_text().replace('"passed": true', '"passed": false'))
        elif case == "parent":
            log = pilot.root / "runs" / "S101-RMS" / "events.jsonl"
            log.write_text(log.read_text().replace("pilot-switch-101-RMS", "something-else", 3))
            pilot = S2M.PilotReference(pilot.root)
        else:
            S2M.write_json(state / "runs" / "S104-RMS" / "failure.json", {"type": "numerical", "message": "nan"})
        report = S2M.build_report(state, pilot, docs)
        assert report["h2"]["decision"]["category"] == "INVALID OR INCOMPLETE", case
        assert report["r2"]["decision"]["category"] == "INVALID OR INCOMPLETE", case


def test_progress_accounting(tmp_path):
    state = tmp_path / "s2state"
    S2.write_json(state / "runs" / "S101-RMS" / "completion.json", {})
    S2.write_json(state / "runs" / "S104-RMS" / "latest.pt.json", {"progress": {"stage": "python", "step": 10}})
    remaining = S2.remaining_updates(state, "RMS")
    full, branch = S2.PREFIX_END + 3 * S2.CONTINUATION_UPDATES, S2.CONTINUATION_UPDATES
    assert remaining == 2 * branch + (full - (S2.PREFIX_END + branch + 10)) + 2 * full
    assert "  S104-RMS: python step 10/6104" in S2.progress_summary(state)


# =============================================================================
# 4. Data preparation: pilot equivalence, deduplication against the frozen corpus, and the runner contract
# =============================================================================
def test_split_rules_are_the_pilots():
    from domain_shift_forgetting.data.splits import c4_validation_split, python_split
    rng = random.Random(3)
    for _ in range(2000):
        digest = hashlib.sha256(str(rng.random()).encode()).hexdigest()
        assert PD.c4_validation_split(digest) == c4_validation_split(digest)
        assert PD.group_split(digest) == python_split(digest)
    owm = PD.CANDIDATES["openwebmath"]
    assert PD.url_group("https://a.b/c", "x") == PD.url_group(" https://a.b/c ", "y")  # URL wins over content
    assert PD.url_group("", "x") != PD.url_group(None, "y")
    splits = [PD.assign_split(owm, "train", "c", f"https://site{i}.org/p") for i in range(20000)]
    assert abs(splits.count("train") / 20000 - 0.90) < 0.01 and abs(splits.count("test") / 20000 - 0.05) < 0.01
    mc4 = PD.CANDIDATES["mc4-zh"]
    assert PD.assign_split(mc4, "train", "c", None) == "train"
    assert PD.assign_split(mc4, "validation", digest, None) == c4_validation_split(digest)


def test_candidate_file_patterns_match_the_pinned_repositories():
    files = {"mc4-de": ["multilingual/c4-de.tfrecord-00000-of-02048.json.gz",
                        "multilingual/c4-de-validation.tfrecord-00015-of-00016.json.gz"],
             "mc4-zh": ["multilingual/c4-zh.tfrecord-01023-of-01024.json.gz",
                        "multilingual/c4-zh-validation.tfrecord-00001-of-00002.json.gz"],
             "mc4-ru": ["multilingual/c4-ru.tfrecord-04095-of-04096.json.gz",
                        "multilingual/c4-ru-validation.tfrecord-00000-of-00032.json.gz"]}
    decoys = ["multilingual/c4-zh-Latn.tfrecord-00000-of-00008.json.gz", "multilingual/c4-ru-Latn.tfrecord-00000-"
              "of-00256.json.gz", "en/c4-train.00000-of-01024.json.gz", "multilingual/c4-de.tfrecord-00000-of-02048.json"]
    for cid, (train, validation) in files.items():
        found = PD.upstream_files(PD.CANDIDATES[cid], [train, validation, *decoys])
        assert found == {"train": [train], "validation": [validation]}, cid
    owm = "data/train-00000-of-00114-5a023365406cb9c4.parquet"
    assert PD.upstream_files(PD.CANDIDATES["openwebmath"], [owm, "README.md", "imgs/x.png"]) == {"train": [owm]}
    assert PD.shard_order(["b", "a", "c"]) == sorted(["b", "a", "c"],
                                                     key=lambda p: hashlib.sha256(f"20260911:{p}".encode()).hexdigest())
    assert PD.CANDIDATES["mc4-de"].revision == PD.C4_REVISION == \
        json.loads((ROOT / "reports/h1-kaggle/h1state/config.json").read_text())["data"]["sources"]["web"]["revision"]


class FakeEncoder:
    """Bytes as token ids (no tokenizer download in tests)."""

    def encode_ordinary(self, text):
        return list(text.encode("utf-8"))

    def encode_ordinary_batch(self, texts, num_threads=1):
        return [self.encode_ordinary(t) for t in texts]


def _gz_shard(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def test_pool_filling_is_deterministic_and_stops_per_document(tmp_path, monkeypatch):
    candidate = PD.CANDIDATES["mc4-de"]
    shards = {"train": ["t1.json.gz", "t2.json.gz"], "validation": ["v1.json.gz"]}
    rng = random.Random(5)
    for name in shards["train"] + shards["validation"]:
        _gz_shard(tmp_path / name, [{"text": "" if i % 7 == 0 else "w" * rng.randint(50, 150),
                                     "url": f"https://x/{name}/{i}"} for i in range(60)])
    monkeypatch.setattr(PD, "download", lambda c, path, cache, expected=None: tmp_path / path)

    readers = {u: PD.RowReader(candidate, u, paths, tmp_path, {}) for u, paths in shards.items()}
    db, raw, sequence = PD.open_pool(tmp_path / "pool.sqlite"), Counter(), [0]
    PD.fill_pool(db, candidate, readers, FakeEncoder(), {"train": 3000, "dev": 300, "test": 300}, raw, sequence, 7)
    assert raw["train"] >= 3000 and raw["dev"] >= 300 and raw["test"] >= 300
    rows = db.execute("SELECT seq, upstream, split, metadata FROM docs ORDER BY seq").fetchall()
    trains = [json.loads(r[3])["row"] for r in rows if r[1] == "train"]
    assert trains == [i for i in range(60) if i % 7][:len(trains)]  # file order, empty texts skipped
    assert raw["train"] - 3000 < 152  # stopped at the first document that reached the target
    assert {r[2] for r in rows if r[1] == "validation"} <= {"dev", "test"}
    # Raising a target continues exactly where the previous fill stopped (no row read twice or skipped).
    before = len(trains)
    PD.fill_pool(db, candidate, readers, FakeEncoder(), {"train": 9000, "dev": 300, "test": 300}, raw, sequence, 7)
    more = [json.loads(r[0])["row"] for r in db.execute("SELECT metadata FROM docs WHERE upstream='train' ORDER BY seq")]
    expected = [(s, i) for s in shards["train"] for i in range(60) if i % 7]
    got = [(json.loads(r[0])["shard"], json.loads(r[0])["row"])
           for r in db.execute("SELECT metadata FROM docs WHERE upstream='train' ORDER BY seq")]
    assert got == expected[:len(got)] and len(more) > before and readers["train"].consumed_shards == shards["train"]
    assert readers["train"].dropped["empty_text"] >= 8


def test_probe_sample_is_deterministic_and_never_reads_held_out_documents(tmp_path, monkeypatch):
    owm = PD.CANDIDATES["openwebmath"]
    listing = ["data/train-00001-of-00114-aa.parquet", "data/train-00000-of-00114-bb.parquet"]
    rows = [(i, f"document {i} " * 40, f"https://site{i}.org/page", {}) for i in range(400)]
    monkeypatch.setattr(PD, "iter_streamed_rows", lambda c, path, cache: iter(rows))
    tokens, info = PD.probe_sample(owm, FakeEncoder(), tmp_path, listing, windows=8)
    assert len(tokens) == 8 * 512 + 1 and info["documents"][0]["shard"] == PD.shard_order(listing)[0]
    for doc in info["documents"]:
        assert PD.group_split(PD.url_group(f"https://site{doc['row']}.org/page", "")) == "train"
    again, _ = PD.probe_sample(owm, FakeEncoder(), tmp_path, listing, windows=8)
    assert np.array_equal(tokens, again) and info["tokens_sha256"] == hashlib.sha256(tokens.tobytes()).hexdigest()


def _pool_db(path: Path, docs, schema: str) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    if schema == "pilot":
        db.execute("CREATE TABLE docs(id TEXT PRIMARY KEY, domain TEXT, upstream TEXT, content_hash TEXT, tokens BLOB, "
                   "metadata TEXT)")
        for doc_id, upstream, split, digest, tokens in docs:
            db.execute("INSERT INTO docs VALUES (?,?,?,?,?,?)", (doc_id, "web", upstream, digest, tokens.tobytes(),
                                                                 json.dumps({"aliases": [], "content_sha256": digest})))
    else:
        db.execute("CREATE TABLE docs(id TEXT PRIMARY KEY, seq INTEGER, upstream TEXT, split TEXT, content_hash TEXT, "
                   "tokens BLOB, metadata TEXT)")
        for index, (doc_id, upstream, split, digest, tokens) in enumerate(docs):
            db.execute("INSERT INTO docs VALUES (?,?,?,?,?,?,?)", (doc_id, index, upstream, split, digest,
                                                                   tokens.tobytes(),
                                                                   json.dumps({"content_sha256": digest})))
    db.commit()
    return db


def _synthetic_documents(rng, count, upstream, prefix):
    docs = []
    for i in range(count):
        tokens = rng.integers(0, 50000, size=int(rng.integers(150, 400))).astype("<u2")
        digest = hashlib.sha256(tokens.tobytes()).hexdigest()
        split = "train" if upstream == "train" else PD.c4_validation_split(digest)
        docs.append((f"{prefix}{i:04d}", upstream, split, digest, tokens))
    return docs


def test_deduplication_and_packing_equal_the_pilots_without_frozen_references(tmp_path):
    from domain_shift_forgetting import pilot_data
    rng = np.random.default_rng(9)
    docs = _synthetic_documents(rng, 120, "train", "t") + _synthetic_documents(rng, 80, "validation", "v")
    duplicate = docs[130][4].copy()  # an exact copy of a held-out document inside train
    near = docs[140][4].copy()
    near[50] = (near[50] + 1) % 50000  # a near copy (one token changed)
    docs += [("t9998", "train", "train", docs[130][3], duplicate),
             ("t9999", "train", "train", hashlib.sha256(near.tobytes()).hexdigest(), near)]
    theirs = pilot_data.filter_pool(_pool_db(tmp_path / "a.sqlite", docs, "pilot"), tmp_path / "a.jsonl")
    mine, audit = PD.filter_pool(_pool_db(tmp_path / "b.sqlite", docs, "s2"), None, tmp_path / "b.jsonl")
    for split in ("train", "dev", "test"):
        assert sorted(mine[f"x_{split}"]) == sorted(theirs[f"web_{split}"]), split
    assert audit["exact_removed"] == 1 and audit["near_removed"] == 1 and "t9998" not in mine["x_train"]
    quotas = {"x_train": 20 * 512 + 1, "x_dev": 3000, "x_test": 3000}
    pilot_quotas = {f"web_{k[2:]}": v for k, v in quotas.items()}
    pilot_data.materialize(_pool_db(tmp_path / "c.sqlite", docs, "pilot"),
                           {f"web_{k[2:]}": v for k, v in mine.items()}, tmp_path / "pilot-out", {}, {}, pilot_quotas)
    PD.materialize(_pool_db(tmp_path / "d.sqlite", docs, "s2"), mine, tmp_path / "s2-out", quotas)
    for split in ("train", "dev", "test"):
        folder = ("reserved-test", "reserved-test") if split == "test" else ("online", "s2online")
        for suffix in (".bin", ".owners.bin"):
            assert (tmp_path / "pilot-out" / folder[0] / f"web_{split}{suffix}").read_bytes() == \
                (tmp_path / "s2-out" / folder[1] / f"x_{split}{suffix}").read_bytes(), (split, suffix)
        theirs_docs = json.loads((tmp_path / "pilot-out" / folder[0] / f"web_{split}.documents.json").read_text())
        mine_docs = json.loads((tmp_path / "s2-out" / folder[1] / f"x_{split}.documents.json").read_text())
        keys = ("id", "token_start", "tokens", "truncated")
        assert [{k: d[k] for k in keys} for d in mine_docs["documents"]] == \
            [{k: d[k] for k in keys} for d in theirs_docs["documents"]]


def _synthetic_pilot_corpus(root: Path, rng, *, web_windows: int, python_windows: int, dev_windows: int):
    """A tiny corpus with the pilot's layout (online/ + orders/), its orders made by the pilot's generator."""
    online, orders = root / "pilot-dataset" / "online", root / "pilot-dataset" / "orders"
    online.mkdir(parents=True)
    (online / "classes.json").write_text(json.dumps({"classes": CLASSES.tolist()}))
    splits = {}
    for key, n_tokens in (("web_train", web_windows * 512 + 1), ("python_train", python_windows * 512 + 1),
                          ("web_dev", dev_windows * 512 + 1), ("python_dev", dev_windows * 512 + 1)):
        values, owners, docs, count = [], [], [], 0
        while count < n_tokens:
            body = rng.integers(0, 50000, size=299).astype("<u2")
            full = np.concatenate([body, np.array([PD.EOS], dtype="<u2")])
            take = min(len(full), n_tokens - count)
            values.append(full[:take])
            owners.append(np.full(take, len(docs), dtype="<u4"))
            docs.append({"id": f"{key}-{len(docs)}", "token_start": count, "tokens": take, "truncated": take < len(full),
                         "content_sha256": hashlib.sha256(body.tobytes()).hexdigest()})
            count += take
        np.concatenate(values).tofile(online / f"{key}.bin")
        np.concatenate(owners).tofile(online / f"{key}.owners.bin")
        PD.json_write(online / f"{key}.documents.json", {"documents": docs})
        splits[key] = {"tokens": n_tokens, "labels": n_tokens - 1, "file": f"{key}.bin",
                       "sha256": PD.sha256_file(online / f"{key}.bin"), "owners": f"{key}.owners.bin",
                       "owners_sha256": PD.sha256_file(online / f"{key}.owners.bin"),
                       "documents": f"{key}.documents.json",
                       "documents_sha256": PD.sha256_file(online / f"{key}.documents.json")}
    PD.json_write(online / "manifest.json", {
        "schema": "pilot-arrays-v1", "splits": splits, "sources": {"web": {"repo": "allenai/c4", "revision": "r"}},
        "tokenizer_hashes": {"id_to_bytes_sha256": PD.PILOT_ID_TO_BYTES_SHA256},
        "classes_sha256": PD.sha256_file(online / "classes.json"),
        "reserved_test_sealed": {"labels_per_domain": 8388608, "online_path_included": False}})
    orders.mkdir()
    web = np.fromfile(online / "web_train.bin", dtype="<u2")
    manifest = {"array_manifest_sha256": PD.sha256_file(online / "manifest.json"), "seeds": {}}
    for seed in PD.PILOT_SEEDS:
        manifest["seeds"][str(seed)] = {}
        for key, values in PD.pilot_orders(seed, web, web_windows, python_windows).items():
            np.save(orders / f"{seed}-{key}.npy", values, allow_pickle=False)
            manifest["seeds"][str(seed)][key] = {"file": f"{seed}-{key}.npy",
                                                 "sha256": PD.sha256_file(orders / f"{seed}-{key}.npy")}
    PD.json_write(orders / "manifest.json", manifest)
    return online, orders


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """A synthetic pilot corpus + Study 2's prepare_domain output for it (mini-protocol sizes), and the package."""
    root = tmp_path_factory.mktemp("prepared")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(PD, "PREFIX_END", S2M.PREFIX_END)
        patch.setattr(PD, "CONTINUATION_UPDATES", S2M.CONTINUATION_UPDATES)
        rng = np.random.default_rng(13)
        inputs = root / "inputs"
        pilot_online, pilot_orders = _synthetic_pilot_corpus(inputs, rng, web_windows=1100, python_windows=400,
                                                             dev_windows=S2M.FULL_WINDOWS)
        frozen_docs = json.loads((pilot_online / "web_train.documents.json").read_text())["documents"]
        dev_tokens = np.fromfile(pilot_online / "web_dev.bin", dtype="<u2")
        dev_doc = json.loads((pilot_online / "web_dev.documents.json").read_text())["documents"][3]
        near_dev = dev_tokens[dev_doc["token_start"]:dev_doc["token_start"] + dev_doc["tokens"] - 1].copy()
        near_dev[100] = (near_dev[100] + 1) % 50000
        near_digest = hashlib.sha256(near_dev.tobytes()).hexdigest()
        docs = _synthetic_documents(rng, 800, "train", "t") + _synthetic_documents(rng, 300, "validation", "v")
        held_out = [d for d in docs if d[2] in ("dev", "test")]
        docs += [("a-exact-frozen", "train", "train", frozen_docs[5]["content_sha256"],
                  rng.integers(0, 9, 300).astype("<u2")),
                 ("a-near-frozen-dev", "validation", PD.c4_validation_split(near_digest), near_digest, near_dev),
                 ("a-exact-heldout", "train", "train", held_out[0][3], held_out[0][4].copy())]
        db = _pool_db(root / "pool.sqlite", docs, "s2")
        retained, audit = PD.filter_pool(db, PD.FrozenReference(pilot_online), root / "removals.jsonl")
        quotas = {"x_train": 384 * 512 + 1, "x_dev": S2M.FULL_WINDOWS * 512 + 1, "x_test": 4000}
        output = root / "prepared"
        (output / "s2upstream").mkdir(parents=True)
        PD.json_write(output / "s2upstream" / "source-evidence.json", {"status": "captured"})
        selection = root / "selection.json"
        PD.json_write(selection, {"schema": PD.SELECTION_SCHEMA, "status": "selected", "selected": "mc4-zh",
                                  "ranking": ["mc4-zh", "mc4-ru", "mc4-de", "openwebmath"],
                                  "candidates": {cid: {"revision": PD.CANDIDATES[cid].revision, "S": s} for cid, s in
                                                 (("mc4-zh", 0.4), ("mc4-ru", 0.3), ("mc4-de", 0.2),
                                                  ("openwebmath", 0.1))},
                                  "python_reference": {"S": 0.11}})
        PD.finalize(db, retained, audit, selection_path=selection, pilot_online=pilot_online,
                    pilot_orders_root=pilot_orders, output=output, quotas=quotas,
                    tokenizer={"encoding": "gpt2", "id_to_bytes_sha256": PD.PILOT_ID_TO_BYTES_SHA256},
                    provenance={"shards_consumed": {}, "shard_sha256": {}, "dropped_rows": {},
                                "raw_tokens_before_dedup": {}, "filter_rounds": 1})
        acceptance = PD.verify(output, pilot_online, quotas)
        package = PD.package(output, inputs / "study2-dataset", "danny00/test-dataset")
    return {"root": root, "inputs": inputs, "pilot_online": pilot_online, "pilot_orders": pilot_orders,
            "s2_online": package / "s2online", "s2_orders": package / "s2orders", "package": package,
            "output": output, "selection": selection, "audit": audit, "acceptance": acceptance}


def test_prepared_arrays_satisfy_the_runner_contract(prepared, tmp_path):
    """prepare_domain's output (dedup -> pack -> orders -> verify -> package) is exactly what s2_run accepts."""
    audit, package = prepared["audit"], prepared["package"]
    assert (audit["exact_frozen_removed"], audit["near_frozen_dev_removed"], audit["exact_removed"]) == (1, 1, 1)
    order_manifest = json.loads((prepared["output"] / "s2orders" / "manifest.json").read_text())
    assert all(all(v.values()) for v in order_manifest["reproduction_of_pilot_orders"].values())
    assert prepared["acceptance"]["status"] == "accepted"
    assert not list(package.rglob("x_test*")) and (package / "dataset-metadata.json").exists()
    # The runner, in its mini protocol, finds and accepts the package unchanged.
    S2M.write_json(prepared["inputs"] / "pilot-run" / "h1state" / "H1-STATE.json", {"experiment_id": "x"})
    found = S2M.locate_inputs(prepared["inputs"])
    assert found["s2_online"] == prepared["s2_online"].resolve()
    assert found["pilot_online"] == prepared["pilot_online"].resolve()
    (prepared["inputs"] / "pilot-run").rename(tmp_path / "moved-pilot-run")  # keep the shared inputs clean
    identity = S2M.verify_data(found["pilot_online"], found["pilot_orders"], found["s2_online"], found["s2_orders"])
    assert identity["selection_sha256"] == PD.sha256_file(prepared["selection"])
    data = S2M.Data(found["pilot_online"], found["pilot_orders"], found["s2_online"], found["s2_orders"])
    assert set(data.seed_orders(101)) == {"web", "python", "rare", "x"} == set(data.seed_orders(104))
    assert S2M.domain_identity(found["s2_online"])["id"] == "mc4-zh"
    rows = data.train["x"].take(np.asarray(data.seed_orders(104)["x"][:4], dtype=np.int64))
    assert rows.shape == (4, 513) and rows.max() < 50257
    copy = shutil.copytree(package, tmp_path / "tampered")
    tampered = bytearray((copy / "s2online" / "x_train.bin").read_bytes())
    tampered[10] ^= 1
    (copy / "s2online" / "x_train.bin").write_bytes(bytes(tampered))
    with pytest.raises(ValueError, match="Hash mismatch"):
        S2M.verify_data(found["pilot_online"], found["pilot_orders"], copy / "s2online", copy / "s2orders")
    # Only the selection (rank 0) or the pre-registered fallback (rank 1) is accepted, and it must be the arrays' domain.
    assert PD.load_selection(prepared["selection"], 1)[1].id == "mc4-ru"
    for rank, domain, accepted in ((1, "mc4-ru", True), (1, "mc4-zh", False), (2, "mc4-de", False)):
        other = shutil.copytree(package, tmp_path / f"rank{rank}-{domain}")
        manifest = json.loads((other / "s2online" / "manifest.json").read_text())
        manifest["selection"]["rank"], manifest["domain"]["id"] = rank, domain
        PD.json_write(other / "s2online" / "manifest.json", manifest)
        orders = json.loads((other / "s2orders" / "manifest.json").read_text())
        orders["x_array_manifest_sha256"] = PD.sha256_file(other / "s2online" / "manifest.json")
        PD.json_write(other / "s2orders" / "manifest.json", orders)
        if accepted:
            S2M.verify_data(found["pilot_online"], found["pilot_orders"], other / "s2online", other / "s2orders")
        else:
            with pytest.raises(ValueError, match="selection"):
                S2M.verify_data(found["pilot_online"], found["pilot_orders"], other / "s2online", other / "s2orders")


# ---------------------------------------------------------------- the worker and orchestrator, without training
def _fake_ce(domain: str, role: str, completed: int) -> float:
    return {"web": 4.5, "python": 8.0, "x": 9.0}[domain] + (0.01 if role == "quick" else 0.0) - 1e-3 * completed


def _fake_energy(domain: str, completed: int) -> float:
    return {"web": 1.0, "python": 0.8, "x": 0.5}[domain] * (1.0 + 1e-3 * completed)


def _fake_diag_measurement(domains, completed: int) -> dict:
    return {"sites": {f"{site}.h": {"all": {d: {"mean_squared_norm": _fake_energy(d, completed)} for d in domains}}
                      for site in S2.SITES}}


def fake_pilot_state(root: Path, real_switch: tuple = (101, "RMS")) -> Path:
    """A mini 'pilot' h1state: real h1-ckpt-1 switch state for one run, hash-consistent placeholders for the rest, and
    the prefix-end records the lineage check compares against."""
    state = root / "h1state"
    S2M.write_json(state / "config.json", {"config_sha256": "fake-pilot-config"})
    S2M.write_json(state / "H1-STATE.json", {"experiment_id": "fake-pilot"})
    for seed in S2M.PILOT_SEEDS:
        for condition in S2M.CONDITIONS:
            run = state / "runs" / f"S{seed}-{condition}"
            run.mkdir(parents=True)
            if (seed, condition) == real_switch:
                model, optimizer, scaler = H1M.make_state(seed, condition, CPU)
                model.completed_updates.fill_(S2M.PREFIX_END)
                digest = H1M.save_checkpoint(run / "switch.pt", model, optimizer, scaler,
                                             {"stage": "prefix", "step": S2M.PREFIX_END},
                                             {"config_sha256": "fake-pilot-config", "seed": seed, "condition": condition})
            else:
                (run / "switch.pt").write_bytes(f"placeholder {seed} {condition}".encode())
                digest = hashlib.sha256((run / "switch.pt").read_bytes()).hexdigest()
                S2M.write_json(run / "switch.pt.json", {"sha256": digest})
            S2M.write_json(run / "completion.json", {"switch_sha256": digest})
            rows = [_record(S2M, "prefix", S2M.PREFIX_END, role, domain, _fake_ce(domain, role, S2M.PREFIX_END), None)
                    for role in ("full", "quick") for domain in ("web", "python")]
            rows.append(_diag(S2M, "prefix", S2M.PREFIX_END, "both", {d: _fake_energy(d, S2M.PREFIX_END)
                                                                         for d in ("web", "python")}))
            _write(run / "events.jsonl", rows)
    return state


@pytest.fixture
def no_training(monkeypatch):
    """Replace the GPU work with cheap stand-ins whose outputs depend only on the model's update count, and make any
    attempt to start a worker process fail loudly (the user's PC has a GPU)."""
    calls = {"updates": []}

    def fake_update(model, optimizer, scaler, stream, indices, microbatch, device):
        assert len(indices) == 32 and max(indices) < stream.windows  # orders and offsets stay in range
        u = int(model.completed_updates) + 1
        model.completed_updates.fill_(u)
        calls["updates"].append(u)
        if calls.get("stop_at") == u:  # one interruption (u recurs in every branch: each restarts at the switch)
            calls["stop_flag"][0] = True
            calls["stop_at"] = None
        return {"u": u, "ce": 1.0, "gn": 0.5, "clip": False, "retry": 0, "scale": None}

    def fake_evaluate(model, data, domain, role, rare, eval_microbatch, device):
        completed = int(model.completed_updates)
        return _record(S2M, "?", 0, role, domain, _fake_ce(domain, role, completed), None)

    def fake_diagnostics(model, probes, class_letters, rare, microbatch, device):
        return _fake_diag_measurement(list(probes), int(model.completed_updates))

    def no_process(*args, **kwargs):
        raise AssertionError("tests must never start a process")

    monkeypatch.setattr(S2M, "ordered_update", fake_update)
    monkeypatch.setattr(S2M, "evaluate", fake_evaluate)
    monkeypatch.setattr(S2M, "diagnostics", fake_diagnostics)
    monkeypatch.setattr(S2M, "gpu_count", lambda: 0)
    monkeypatch.setattr(S2M.subprocess, "Popen", no_process)
    return calls


def test_worker_writes_exactly_the_records_the_report_reads(prepared, tmp_path, no_training):
    pilot_dir = fake_pilot_state(tmp_path / "pilot")
    pilot = S2M.PilotReference(pilot_dir)
    pilot.verify()
    data = S2M.Data(prepared["pilot_online"], prepared["pilot_orders"], prepared["s2_online"], prepared["s2_orders"])
    state = tmp_path / "s2state"
    cfg = S2M.config_document(8, 4, 16, pilot.pins, S2M.domain_identity(prepared["s2_online"])) | {
        "config_sha256": "test-config", "checkpoint_seconds": 10 ** 9}
    S2M.write_json(state / "config.json", cfg)
    deadline = 4e9
    # A pilot seed: X branch only, restored from the pilot's switch state, lineage-checked against its records.
    flag = [False]
    assert S2M.run_job(101, "RMS", state, data, cfg, pilot, deadline, flag, CPU) == "complete"
    # A fresh seed, interrupted inside its web branch and resumed from latest.pt.
    no_training["stop_at"], no_training["stop_flag"] = S2M.PREFIX_END + 5, flag
    with pytest.raises(S2M.Stop):
        S2M.run_job(104, "Taper-minus", state, data, cfg, pilot, deadline, flag, CPU)
    progress = json.loads((state / "runs" / "S104-Taper-minus" / "latest.pt.json").read_text())["progress"]
    assert (progress["stage"], progress["step"]) == ("web", 5)
    flag[0] = False
    assert S2M.run_job(104, "Taper-minus", state, data, cfg, pilot, deadline, flag, CPU) == "complete"
    C, P = S2M.CONTINUATION_UPDATES, S2M.PREFIX_END
    assert no_training["updates"] == list(range(P + 1, P + C + 1)) + list(range(1, P + 1)) + \
        list(range(P + 1, P + C + 1)) * 3  # global clock: each branch restarts at the switch
    train_rows = [json.loads(line) for line in (state / "runs" / "S104-Taper-minus" / "train.jsonl").read_text().splitlines()]
    assert [(r["stage"], r["step"]) for r in train_rows] == [("prefix", s) for s in range(1, P + 1)] + \
        [(b, s) for b in ("web", "python", "x") for s in range(1, C + 1)]
    for seed, condition in ((101, "RMS"), (104, "Taper-minus")):
        run = state / "runs" / f"S{seed}-{condition}"
        completion = json.loads((run / "completion.json").read_text())
        assert completion["stages"] == list(S2M.job_stages(seed)) and not (run / "latest.pt").exists()
        assert all((run / f"{stage}-final.pt").exists() for stage in S2M.job_stages(seed) if stage != "prefix")
    assert json.loads((state / "runs" / "S101-RMS" / "completion.json").read_text())["switch_sha256"] == \
        pilot.switch_sha256(101, "RMS")
    records = S2M.Records(state, pilot)
    for seed, condition in ((101, "RMS"), (104, "Taper-minus")):
        for stage, step, role, domain in S2M.expected_records(seed):
            if records.source(seed, stage) == "s2":
                assert records.get(seed, condition, stage, step, role, domain) is not None, (seed, stage, step, role)
        for stage in S2M.job_stages(seed):
            for step in (S2M.DIAGNOSTIC_CONTINUATION if stage != "prefix" else (P,)):
                assert records.diag(seed, condition, stage, step) is not None
            if stage != "prefix":
                lineage = records.s2[seed, condition].get(stage, 0, "lineage", "switch")
                assert lineage["passed"] and lineage["max_ce_diff"] == 0.0 and lineage["diag_compared"]
    assert records.missing == [] and records.problems == []
    shutil.rmtree(state, ignore_errors=True)


def test_worker_stops_on_a_switch_state_that_does_not_reproduce_its_records(prepared, tmp_path, no_training):
    pilot_dir = fake_pilot_state(tmp_path / "pilot")
    log = pilot_dir / "runs" / "S101-RMS" / "events.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    for row in rows:
        if row["role"] == "full" and row["domain"] == "web":
            row["ce"] += 0.01
            row["total"][0] += 0.01 * row["total"][1]
            row["classes"]["W"][0] += 0.01 * row["total"][1]
            row["doc_sums"] = [row["total"][0] / 8] * 8
    _write(log, rows)
    pilot = S2M.PilotReference(pilot_dir)
    data = S2M.Data(prepared["pilot_online"], prepared["pilot_orders"], prepared["s2_online"], prepared["s2_orders"])
    cfg = S2M.config_document(8, 4, 16, pilot.pins, S2M.domain_identity(prepared["s2_online"])) | {
        "config_sha256": "test-config", "checkpoint_seconds": 10 ** 9}
    with pytest.raises(S2M.LineageError):
        S2M.run_job(101, "RMS", tmp_path / "s2state", data, cfg, pilot, 4e9, [False], CPU)
    assert no_training["updates"] == []  # nothing trained on an unverified state


def test_orchestrator_bootstraps_refuses_and_freezes_its_configuration(prepared, tmp_path, monkeypatch, no_training):
    pilot_dir = fake_pilot_state(tmp_path / "pilot")
    data_args = ["--pilot-online", str(prepared["pilot_online"]), "--pilot-orders", str(prepared["pilot_orders"]),
                 "--s2-online", str(prepared["s2_online"]), "--s2-orders", str(prepared["s2_orders"]),
                 "--pilot-state", str(pilot_dir)]
    empty = tmp_path / "empty"
    empty.mkdir()

    def main(inputs, work):
        return S2M.main(S2M.parse(["main", "--mini", "--inputs", str(inputs), "--work", str(work), *data_args]))

    monkeypatch.setenv("S2_SESSION_HOURS", "0.5")
    monkeypatch.delenv("S2_ALLOW_FRESH_START", raising=False)
    assert main(empty, tmp_path / "w1") == 0 and (tmp_path / "w1" / "S2-BOOTSTRAP.json").exists()  # CPU bootstrap
    monkeypatch.setattr(S2M, "gpu_count", lambda: 1)
    assert main(empty, tmp_path / "w2") == 4  # never silently starts a new experiment
    monkeypatch.setattr(S2M, "gpu_count", lambda: 0)
    assert main(tmp_path / "w1", tmp_path / "w3") == 6  # authorized fresh start, config frozen, then "no GPU"
    config = json.loads((tmp_path / "w3" / "s2state" / "config.json").read_text())
    assert config["config_sha256"] == S2M.digest_json({k: v for k, v in config.items()
                                                       if k not in ("config_sha256", "checkpoint_seconds")})
    assert config["domain"]["id"] == "mc4-zh" and config["pilot"]["config_sha256"] == "fake-pilot-config"
    assert config["data"]["selection_sha256"] == PD.sha256_file(prepared["selection"])
    assert (tmp_path / "w3" / "s2state" / "decision-report.json").exists()
    assert main(tmp_path / "w3", tmp_path / "w4") == 6  # resumes the same frozen configuration
    monkeypatch.setitem(S2M.DECISION_THRESHOLDS, "small_d", 0.02)
    assert main(tmp_path / "w4", tmp_path / "w5") == 5  # changed code/config: refuses to mix strata


def test_x_orders_are_independent_permutations():
    a, b = PD.x_order(104, 300_000), PD.x_order(105, 300_000)
    needed = PD.CONTINUATION_UPDATES * 32
    assert len(a) == needed and len(np.unique(a)) == needed and a.max() < 300_000 and a.dtype == np.dtype("<u8")
    assert not np.array_equal(a, b) and np.array_equal(a, PD.x_order(104, 300_000))
    assert not np.array_equal(a[:1000], np.random.default_rng(104).permutation(300_000)[:1000])
    with pytest.raises(ValueError):
        PD.x_order(101, needed - 1)


@needs_data
def test_order_generator_reproduces_the_pilots_seed_101_permutations():
    manifest = json.loads((DATA / "online" / "manifest.json").read_text())
    web_windows = (manifest["splits"]["web_train"]["tokens"] - 1) // 512
    python_windows = (manifest["splits"]["python_train"]["tokens"] - 1) // 512
    rng = np.random.default_rng(101)  # the first two draws of pilot_orders(); rare flags are the heavy test below
    web = rng.permutation(web_windows)[:(PD.PREFIX_END + PD.CONTINUATION_UPDATES) * 32].astype("<u8")
    code = rng.permutation(python_windows)[:PD.CONTINUATION_UPDATES * 32].astype("<u8")
    assert np.array_equal(web, np.load(DATA / "orders" / "101-web.npy"))
    assert np.array_equal(code, np.load(DATA / "orders" / "101-python.npy"))


@heavy
@needs_data
def test_order_generator_reproduces_the_pilots_files_including_rare_flags(tmp_path):
    manifest = json.loads((DATA / "online" / "manifest.json").read_text())
    web = np.memmap(DATA / "online" / "web_train.bin", dtype="<u2", mode="r")
    orders = PD.pilot_orders(101, web, (len(web) - 1) // 512, (manifest["splits"]["python_train"]["tokens"] - 1) // 512)
    pilot = json.loads((DATA / "orders" / "manifest.json").read_text())["seeds"]["101"]
    for key, values in orders.items():
        assert PD._saved_sha256(values, tmp_path / "x.npy") == pilot[key]["sha256"], key


def test_protocol_states_the_constants_the_code_uses():
    """study2/PROTOCOL.md is what gets registered: it must quote the code's pins and constants exactly."""
    text = (ROOT / "study2" / "PROTOCOL.md").read_text(encoding="utf-8")
    pins = S2.PILOT_PINS
    for value in [pins["config_sha256"], pins["experiment_id"], pins["runner_sha256"], pins["online_manifest_sha256"],
                  pins["orders_manifest_sha256"], PD.PILOT_ID_TO_BYTES_SHA256, PD.C4_REVISION, PD.OWM_REVISION,
                  *pins["switch_sha256"].values(), *pins["events_digest"].values()]:
        assert value in text, value
    for value in ("20260911", str(PD.X_ORDER_ENTROPY), str(S2.ANALYSIS["bootstrap_seed"]), "3,300,261,888",
                  "110,000,129", "214,844", "195,328", "488,320", "2,097,153", "8,388,609", "0.1099", "2.571", "4.303",
                  "2.015", "2.920", "±0.015", "1.15", "10% of its quota", "12 expansions", "10⁻³", "B = 10,000"):
        assert value in text, value
    assert S2.ANALYSIS["t95"] == {"3": 4.303, "6": 2.571} and S2.ANALYSIS["t90"] == {"3": 2.920, "6": 2.015}
    assert (PD.RAW_TARGET_MARGIN, PD.RAW_TARGET_STEP, PD.MAX_FILTER_ROUNDS) == (0.15, 0.10, 12)
    assert S2.LINEAGE_CE_TOLERANCE == S2.LINEAGE_ENERGY_TOLERANCE == PROBE.REPRODUCTION_TOLERANCE == 1e-3
    assert PD.quota_tokens("x_train") == 110_000_129 and (110_000_129 - 1) // 512 == 214_844
    assert [PD.CANDIDATES[c].id for c in PD.CANDIDATE_ORDER] == ["mc4-de", "mc4-ru", "mc4-zh", "openwebmath"]
    for threshold in ("−0.03", "0.03", "0.015", "0.05", "2%"):
        assert threshold in text


# =============================================================================
# 5. Notebooks
# =============================================================================
def test_notebooks_embed_exactly_the_repository_code(tmp_path, monkeypatch):
    monkeypatch.setattr(BUILD, "KERNELS_DIR", tmp_path)
    monkeypatch.setattr(BUILD, "ENVIRONMENT", tmp_path / "environment.json")
    monkeypatch.setattr(BUILD, "REGISTRATION", tmp_path / "registration-manifest.json")
    with pytest.raises(SystemExit):  # no notebook without the frozen registration manifest
        BUILD.build_probe()
    (tmp_path / "registration-manifest.json").write_bytes(b'{\n  "schema": "s2-registration-1"\n}\n')
    folders = {"probe": BUILD.build_probe(), "prepare": BUILD.build_prepare(), "smoke": BUILD.build_smoke()}
    expected_files = {"s2_run.py": HERE / "s2_run.py", "prepare_domain.py": HERE / "prepare_domain.py",
                      "probe_domains.py": HERE / "probe_domains.py", "h1_run.py": ROOT / "kaggle_h1" / "h1_run.py",
                      "registration-manifest.json": tmp_path / "registration-manifest.json"}
    for phase in ("bootstrap", "gpu"):
        folders[f"run-{phase}"] = shutil.copytree(BUILD.build_run(phase), tmp_path / f"copy-{phase}")
    for name, folder in folders.items():
        meta = json.loads((folder / "kernel-metadata.json").read_text())
        notebook = json.loads((folder / meta["code_file"]).read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            text = "".join(cell["source"])
            if cell["cell_type"] != "code":
                continue
            if text.startswith("%%writefile"):
                first, body = text.split("\n", 1)
                assert body == BUILD.source(expected_files[Path(first.split()[1]).name]), (name, first)
            else:
                compile(text, f"{name}-cell", "exec")
        written = ["".join(c["source"]).split("\n", 1)[0] for c in notebook["cells"] if c["cell_type"] == "code"]
        assert "%%writefile /kaggle/working/registration-manifest.json" in written, name
    smoke = json.loads((folders["smoke"] / "s2-single-notebook-smoke.ipynb").read_text(encoding="utf-8"))
    h1_cell = next("".join(c["source"]) for c in smoke["cells"]
                   if "".join(c["source"]).startswith("%%writefile /kaggle/working/h1_run.py")).split("\n", 1)[1]
    assert hashlib.sha256(h1_cell.encode()).hexdigest() == S2.PILOT_PINS["runner_sha256"]
    meta = {n: json.loads((f / "kernel-metadata.json").read_text()) for n, f in folders.items()}
    assert (meta["probe"]["enable_gpu"], meta["probe"]["enable_internet"]) == (True, True)
    assert meta["probe"]["kernel_sources"] == [BUILD.PILOT_RUN]
    assert (meta["prepare"]["enable_gpu"], meta["prepare"]["enable_internet"]) == (False, True)
    assert meta["prepare"]["kernel_sources"] == [f"{BUILD.USER}/{BUILD.PROBE_SLUG}"]
    assert (meta["smoke"]["enable_internet"], meta["run-gpu"]["enable_internet"]) == (False, False)
    assert meta["run-gpu"]["kernel_sources"] == [BUILD.PILOT_RUN, f"{BUILD.USER}/{BUILD.RUN_SLUG}"]
    assert meta["run-bootstrap"]["kernel_sources"] == [] and not meta["run-bootstrap"]["enable_gpu"]
    assert meta["run-gpu"]["dataset_sources"] == [BUILD.PILOT_DATASET, BUILD.S2_DATASET]
    assert not any("docker_image" in m for m in meta.values())
    (tmp_path / "environment.json").write_text(json.dumps({"docker_image": "gcr.io/kaggle-gpu-images/python@sha256:x"}))
    pinned = json.loads((BUILD.build_run("gpu", 4.5) / "kernel-metadata.json").read_text())
    assert pinned["docker_image"] == "gcr.io/kaggle-gpu-images/python@sha256:x"
    capped = (tmp_path / "kernel-run" / f"{BUILD.RUN_SLUG}.ipynb").read_text(encoding="utf-8")
    assert "SESSION_HOURS = 4.5" in capped
    with pytest.raises(ValueError):
        BUILD.build_run("gpu", 12)
    assert "docker_image" not in json.loads((BUILD.build_prepare() / "kernel-metadata.json").read_text())
