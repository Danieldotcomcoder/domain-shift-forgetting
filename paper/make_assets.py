"""Generate every number, table and figure in the paper from the run's own records.

Reads reports/h1-kaggle/ (decision report, config, session record, per-run evaluation logs,
completion receipts, and corpus/{audit,manifest}.json). Recomputes the primary
contrasts independently from the raw evaluation records and asserts agreement with the frozen
analysis output before writing anything. Nothing in the manuscript is typed by hand.

  python paper/make_assets.py
"""
import hashlib
import json
import math
import re
import statistics
import subprocess
import datetime as dt
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "reports" / "h1-kaggle" / "h1state"
ONLINE = ROOT / "reports" / "h1-kaggle" / "corpus"  # copies of the corpus audit + manifest (hashes/counts only)
OUT = ROOT / "paper" / "generated"
FIG = ROOT / "paper" / "figures"
SEEDS = (101, 102, 103)
CONDS = ("RMS", "Taper-minus")
PREFIX_END, CONT = 9156, 6104
PERSIST = (1525, 3050, 6104)
FULL_CONT = sorted({0, 1, 10, 100, 1525, 3050, 6104} | set(range(305, 6101, 305)))
QUICK_CONT = sorted({0, 1, 2, 5, 10, 20, 1525, 3050, 6104} | set(range(50, 1001, 50)) | set(range(1100, 6101, 100)))
FULL_PREFIX = (763, 3052, 6104, 6409, 6714, 7019, 7324, 7629, 7934, 8239, 8544, 8849, 9156)

# Reference palette (dataviz skill), validated: blue = RMS, orange = Taper-minus.
COLOR = {"RMS": "#2a78d6", "Taper-minus": "#eb6834"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"


def load():
    report = json.loads((STATE / "decision-report.json").read_text(encoding="utf-8"))
    config = json.loads((STATE / "config.json").read_text(encoding="utf-8"))
    session = json.loads((STATE / "sessions.jsonl").read_text(encoding="utf-8").strip().splitlines()[0])
    events, completion = {}, {}
    for seed in SEEDS:
        for cond in CONDS:
            run = STATE / "runs" / f"S{seed}-{cond}"
            completion[seed, cond] = json.loads((run / "completion.json").read_text())
            for line in (run / "events.jsonl").read_text().splitlines():
                r = json.loads(line)
                events[seed, cond, r["stage"], r["step"], r["role"], r["domain"]] = r
    return report, config, session, events, completion


def mm(x, digits=4, sign=True):
    return "\\ensuremath{" + (f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}").replace("+", "{+}") + "}"


def learning_rate(u):
    if u <= 305:
        return 0.0006 * u / 305
    return 0.00006 + 0.5 * (0.0006 - 0.00006) * (1 + math.cos(math.pi * (u - 305) / (15260 - 305)))


def second_draft_analyses(ev, rows, config, session):
    """Integrity timeline, per-phase clipping, trajectory windows, R contribution, document
    bootstrap, activation-scale analysis (Finding A) and LSH recall. Returns macros."""
    out = {}
    # ---- integrity: the logged configuration hash covers the thresholds and is recomputable
    body = {k: v for k, v in config.items() if k not in ("config_sha256", "checkpoint_seconds")}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert digest == config["config_sha256"], "config hash does not recompute"
    log = json.loads((ROOT / "reports" / "h1-kaggle" / "h1-single-notebook-run.log").read_text(encoding="utf-8"))
    lines = [line for entry in log for line in entry.get("data", "").splitlines()]

    def stamp(needle):
        hit = next(line for line in lines if needle in line)
        return re.search(r"\[(\d\d:\d\d:\d\d)\]", hit).group(1)

    out["logConfigTime"] = stamp("NEW experiment created")
    assert config["config_sha256"][:16] in next(l for l in lines if "NEW experiment created" in l)
    s101 = sorted(stamp(f"S101-{c} {b}@6104") for c in CONDS for b in ("web", "python"))
    out["seedOneFirst"], out["seedOneLast"] = s101[0][:5], s101[-1][:5]
    commit_iso = subprocess.check_output(["git", "show", "-s", "--format=%cI", "3ee1dc4"], cwd=ROOT, text=True).strip()
    commit_utc = dt.datetime.fromisoformat(commit_iso).astimezone(dt.UTC)
    out["commitTime"] = commit_utc.strftime("%H:%M")
    assert s101[-1][:5] < out["commitTime"], "seed 101 endpoints were logged before the commit"
    commits = subprocess.check_output(["git", "rev-list", "HEAD"], cwd=ROOT, text=True).split()
    for commit in commits:
        blob = subprocess.check_output(["git", "show", f"{commit}:kaggle_h1/h1_run.py"], cwd=ROOT)
        assert hashlib.sha256(blob).hexdigest() == session["code_sha256"], f"runner differs at {commit}"
    runner = (ROOT / "kaggle_h1" / "h1_run.py").read_text(encoding="utf-8").splitlines()
    out["thresholdLine"] = str(next(i for i, l in enumerate(runner, 1) if l.startswith("DECISION_THRESHOLDS = {")))

    # ---- per-phase gradient clipping from the per-update training logs
    train = {}
    for seed in SEEDS:
        for cond in CONDS:
            path = STATE / "runs" / f"S{seed}-{cond}" / "train.jsonl"
            train[seed, cond] = [json.loads(l) for l in path.read_text().splitlines()]
            assert len(train[seed, cond]) == PREFIX_END + 2 * CONT
    phases = [("Prefix, $u \\le 763$ (calibration)", lambda r: r["stage"] == "prefix" and r["u"] <= 763),
              ("Prefix, $764 \\le u \\le 6{,}103$ (gate decaying)", lambda r: r["stage"] == "prefix" and 764 <= r["u"] <= 6103),
              ("Prefix, $6{,}104 \\le u \\le 9{,}156$ (gate zero)", lambda r: r["stage"] == "prefix" and r["u"] >= 6104),
              ("Web branch", lambda r: r["stage"] == "web"), ("Python branch", lambda r: r["stage"] == "python")]
    table = ["\\begin{tabular}{@{}lrcc@{}}", "\\toprule", "Phase & Updates & RMS & Taper-minus \\\\", "\\midrule"]
    phase_means = {}
    for name, keep in phases:
        cells = []
        for cond in CONDS:
            fr = [100 * np.mean([r["clip"] for r in train[s, cond] if keep(r)]) for s in SEEDS]
            phase_means[name, cond] = float(np.mean(fr))
            cells.append(f"{np.mean(fr):.1f}\\% ({min(fr):.1f}--{max(fr):.1f})")
        n = sum(1 for r in train[101, "RMS"] if keep(r))
        table.append(f"{name} & " + f"{n:,}".replace(",", "{,}") + " & " + " & ".join(cells) + " \\\\")
    table += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "table_clipping.tex").write_text("\n".join(table) + "\n", encoding="utf-8")
    branch_means = [phase_means[name, c] for name, _ in phases[3:] for c in CONDS]
    out["clipBranchLo"], out["clipBranchHi"] = f"{min(branch_means):.0f}", f"{max(branch_means):.0f}"
    out["clipGateZeroRMS"] = f"{phase_means[phases[2][0], 'RMS']:.0f}"
    out["clipGateZeroTaper"] = f"{phase_means[phases[2][0], 'Taper-minus']:.0f}"
    out["clipEarlyHi"] = f"{max(phase_means[phases[i][0], c] for i in (0, 1) for c in CONDS):.0f}"
    assert abs(phase_means[phases[0][0], "RMS"] - phase_means[phases[0][0], "Taper-minus"]) < 1e-9
    out["clipCalib"] = f"{phase_means[phases[0][0], 'RMS']:.0f}"
    out["clipDecayRMS"] = f"{phase_means[phases[1][0], 'RMS']:.0f}"
    out["clipDecayTaper"] = f"{phase_means[phases[1][0], 'Taper-minus']:.0f}"
    out["maxLossScale"] = f"{max(r['scale'] for k in train for r in train[k]):,.0f}"

    # ---- trajectory windows (post hoc grouping of the pre-specified full-dev points)
    point_mean = {u: statistics.mean(rows[s]["full_D"][u] for s in SEEDS) for u in FULL_CONT}
    point_sd = {u: statistics.stdev(rows[s]["full_D"][u] for s in SEEDS) for u in FULL_CONT}
    early = [10, 100, 305]
    middle = [u for u in FULL_CONT if 610 <= u <= 4880]
    late = [u for u in FULL_CONT if u >= 5185]
    out["earlyLo"] = mm(min(point_mean[u] for u in early), 3)
    out["earlyHi"] = mm(max(point_mean[u] for u in early), 3)
    out["midMean"] = mm(statistics.mean(point_mean[u] for u in middle), 3)
    out["midMin"] = mm(min(point_mean[u] for u in middle), 3)
    out["midMax"] = mm(max(point_mean[u] for u in middle), 3)
    out["midCount"] = str(len(middle))
    out["lateMean"] = mm(statistics.mean(point_mean[u] for u in late), 3)
    out["lateCount"] = str(len(late))
    mid_sd = statistics.mean(point_sd[u] for u in middle)
    out["midSD"] = mm(mid_sd, 4, sign=False)
    out["finalSD"] = mm(point_sd[CONT], 4, sign=False)
    out["sdRatio"] = f"{mid_sd / point_sd[CONT]:.1f}"
    # Upper bound, rounded up, so "within N%" is true throughout the late window.
    out["lateLRpct"] = str(math.ceil((learning_rate(PREFIX_END + late[0]) / 6e-5 - 1) * 100))
    out["lateNeg"] = str(sum(point_mean[u] < 0 for u in late))
    out["lateNegSeedPoints"] = str(sum(rows[s]["full_D"][u] < 0 for s in SEEDS for u in late))
    out["lateSeedPoints"] = str(len(SEEDS) * len(late))
    out["dipMean"] = mm(point_mean[100], 4)
    out["dipSD"] = mm(point_sd[100], 4, sign=False)

    # ---- pre-specified rare-token (R) contribution at the endpoint (overlaps W/A/P/X)
    rare_rows, r_contrib, r_mean, r_count = [], [], [], []
    for seed in SEEDS:
        g = lambda c, b: ev[seed, c, b, CONT, "full", "web"]["rare"]
        tp, tw, rp, rw = g("Taper-minus", "python"), g("Taper-minus", "web"), g("RMS", "python"), g("RMS", "web")
        assert len({tp[1], tw[1], rp[1], rw[1]}) == 1
        total = ev[seed, "RMS", "web", CONT, "full", "web"]["total"][1]
        diff = (tp[0] - tw[0]) - (rp[0] - rw[0])
        r_count.append(tp[1]); r_mean.append(diff / tp[1]); r_contrib.append(diff / total)
    out["Rlabels"] = f"{min(r_count):,}--{max(r_count):,}"
    out["RmeanLo"], out["RmeanHi"] = mm(min(r_mean), 3), mm(max(r_mean), 3)
    out["RcontribLo"], out["RcontribHi"] = mm(min(r_contrib)), mm(max(r_contrib))
    out["RsharePct"] = f"{max(abs(r_contrib[i] / rows[s]['D']) for i, s in enumerate(SEEDS)) * 100:.0f}"

    # ---- document bootstrap at the endpoint (paired across arms and seeds; conditional on models)
    rng = np.random.default_rng(20261007)
    deltas, counts = [], None
    for seed in SEEDS:
        arr = lambda c, b: np.array(ev[seed, c, b, CONT, "full", "web"]["doc_sums"])
        deltas.append((arr("Taper-minus", "python") - arr("Taper-minus", "web")) - (arr("RMS", "python") - arr("RMS", "web")))
        cnt = np.array(ev[seed, "RMS", "web", CONT, "full", "web"]["doc_counts"])
        counts = cnt if counts is None else counts
        assert (cnt == counts).all()
    deltas = np.array(deltas)[:, counts > 0]
    counts = counts[counts > 0]
    assert np.allclose(deltas.sum(1) / counts.sum(), [rows[s]["D"] for s in SEEDS], atol=1e-9)
    B, n_docs = 10_000, len(counts)
    reps = np.empty((B, len(SEEDS)))
    for start in range(0, B, 500):
        idx = rng.integers(0, n_docs, size=(500, n_docs))
        w = np.stack([np.bincount(row, minlength=n_docs) for row in idx])
        reps[start:start + 500] = (w @ deltas.T) / (w @ counts)[:, None]
    mean_reps = reps.mean(1)
    lo, hi = np.percentile(mean_reps, [2.5, 97.5])
    out["bootB"], out["bootLo"], out["bootHi"] = f"{B:,}", mm(lo), mm(hi)
    out["bootSE"] = mm(mean_reps.std(ddof=1), 4, sign=False)
    out["bootSeedSElo"] = mm(reps.std(0, ddof=1).min(), 4, sign=False)
    out["bootSeedSEhi"] = mm(reps.std(0, ddof=1).max(), 4, sign=False)
    # Compare like with like: seed SD vs per-seed bootstrap SE, and seed SE of the mean vs bootstrap SE of the mean.
    seed_sd = statistics.stdev(rows[s]["D"] for s in SEEDS)
    out["bootRatioSeed"] = f"{seed_sd / reps.std(0, ddof=1).mean():.1f}"
    out["seedSEmean"] = mm(seed_sd / math.sqrt(len(SEEDS)), 4, sign=False)
    out["bootRatioMean"] = f"{seed_sd / math.sqrt(len(SEEDS)) / mean_reps.std(ddof=1):.1f}"
    seed_ci = [np.percentile(reps[:, i], [2.5, 97.5]) for i in range(len(SEEDS))]
    supp = ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
            "Seed & R labels & Mean $D$ on R & R contribution to $D$ & Document-bootstrap 95\\% interval for $D$ \\\\",
            "\\midrule"]
    for i, seed in enumerate(SEEDS):
        supp.append(f"{seed} & {r_count[i]:,} & ".replace(",", "{,}") + f"${r_mean[i]:+.3f}$ & ${r_contrib[i]:+.4f}$ & "
                    f"$[{seed_ci[i][0]:+.4f}, {seed_ci[i][1]:+.4f}]$ \\\\".replace("+", "{+}"))
    supp.append(f"Mean of seeds & & & & $[{lo:+.4f}, {hi:+.4f}]$ \\\\".replace("+", "{+}"))
    supp += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "table_rare_bootstrap.tex").write_text("\n".join(supp) + "\n", encoding="utf-8")

    # ---- Finding A: activation scales at the 12 internal normalizer inputs (diagnostic probes)
    sites = [f"{i}.{b}" for i in range(6) for b in ("attention", "mlp")]
    scale = {}
    for cond in CONDS:
        gap, w_signed, p_signed, w_abs, p_abs, specific = [], [], [], [], [], []
        for seed in SEEDS:
            diag = {(k[2], k[3]): r["measurement"] for k, r in ev.items()
                    if k[0] == seed and k[1] == cond and k[4] == "diag"}
            for site in sites:
                E = lambda when, dom: diag[when]["sites"][f"{site}.h"]["all"][dom]["mean_squared_norm"]
                switch, web_end, py_end = ("prefix", PREFIX_END), ("web", CONT), ("python", CONT)
                gap.append(0.5 * math.log(E(switch, "python") / E(switch, "web")))
                w_signed.append(0.5 * math.log(E(web_end, "web") / E(switch, "web")))
                p_signed.append(0.5 * math.log(E(py_end, "web") / E(switch, "web")))
                specific.append(p_signed[-1] - w_signed[-1])
        # Natural-log scale ratios: |.| rows take the absolute value per seed and site, then the mean over the
        # 36 site-seed pairs; signed rows are means of the signed values (also shown as factors exp(mean)).
        scale[cond] = {"gap": np.mean(np.abs(gap)), "w_abs": np.mean(np.abs(w_signed)),
                       "p_abs": np.mean(np.abs(p_signed)), "w_signed": np.mean(w_signed),
                       "p_signed": np.mean(p_signed), "specific": np.mean(np.abs(specific)),
                       "w_shrink": int(np.sum(np.array(w_signed) < 0)), "p_shrink": int(np.sum(np.array(p_signed) < 0)),
                       "pairs": len(w_signed)}
    tag = {"RMS": "R", "Taper-minus": "T"}
    for cond in CONDS:
        v = scale[cond]
        out[f"scaleGap{tag[cond]}"] = f"{v['gap']:.3f}"
        out[f"scaleGapPct{tag[cond]}"] = f"{(math.exp(v['gap']) - 1) * 100:.0f}"
        out[f"scaleWebAbs{tag[cond]}"] = f"{v['w_abs']:.3f}"
        out[f"scalePyAbs{tag[cond]}"] = f"{v['p_abs']:.3f}"
        out[f"scaleWebSigned{tag[cond]}"] = mm(v["w_signed"], 3)
        out[f"scalePySigned{tag[cond]}"] = mm(v["p_signed"], 3)
        out[f"scaleWebFactor{tag[cond]}"] = f"{math.exp(v['w_signed']):.2f}"
        out[f"scalePyFactor{tag[cond]}"] = f"{math.exp(v['p_signed']):.2f}"
        out[f"scaleSpecific{tag[cond]}"] = f"{v['specific']:.3f}"
        out[f"scaleSpecificPct{tag[cond]}"] = f"{(math.exp(v['specific']) - 1) * 100:.0f}"
        out[f"scaleWebShrink{tag[cond]}"] = str(v["w_shrink"])
        out[f"scalePyShrink{tag[cond]}"] = str(v["p_shrink"])
    out["scalePairs"] = str(scale["RMS"]["pairs"])
    R, T = scale["RMS"], scale["Taper-minus"]
    fmt_signed = lambda x: f"${x:+.3f}$ ($\\times{math.exp(x):.2f}$)".replace("+", "{+}")
    t = ["\\begin{tabular}{@{}lrr@{}}", "\\toprule",
         "Natural-log scale ratio (mean over 12 sites $\\times$ 3 seeds) & RMS & Taper-minus \\\\", "\\midrule",
         f"Code vs.\\ web at the switch, $|\\cdot|$ & {R['gap']:.3f} & {T['gap']:.3f} \\\\",
         f"Web-probe change, web branch, $|\\cdot|$ & {R['w_abs']:.3f} & {T['w_abs']:.3f} \\\\",
         f"Web-probe change, Python branch, $|\\cdot|$ & {R['p_abs']:.3f} & {T['p_abs']:.3f} \\\\",
         f"\\quad signed, web branch (factor) & {fmt_signed(R['w_signed'])} & {fmt_signed(T['w_signed'])} \\\\",
         f"\\quad signed, Python branch (factor) & {fmt_signed(R['p_signed'])} & {fmt_signed(T['p_signed'])} \\\\",
         f"\\quad site--seed pairs that shrank, web / Python branch & {R['w_shrink']} / {R['p_shrink']} of {R['pairs']} & "
         f"{T['w_shrink']} / {T['p_shrink']} of {T['pairs']} \\\\",
         f"Code-specific change (Python minus web branch), $|\\cdot|$ & {R['specific']:.3f} & {T['specific']:.3f} \\\\",
         "\\bottomrule", "\\end{tabular}"]
    (OUT / "table_scales.tex").write_text("\n".join(t) + "\n", encoding="utf-8")

    # ---- LSH recall (32 bands x 4 rows): candidate probability for a pair with Jaccard J
    p = lambda J: 1 - (1 - J ** 4) ** 32
    miss = 1 - p(0.85)
    exponent = math.floor(math.log10(miss))
    out["lshMiss"] = f"\\ensuremath{{{miss / 10 ** exponent:.1f}\\times10^{{{exponent}}}}}"
    out["lshRecallSeventy"] = f"{p(0.70):.4f}"
    audit = json.loads((ONLINE / "audit.json").read_text())
    out["shortDocs"] = str(audit["short_shingle_documents"])
    assert audit["missed_candidate_pairs_found"] == 0  # the manuscript says "none reached 0.85"
    return out


def main():
    report, config, session, ev, completion = load()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    def ce(seed, cond, stage, step, role="full", dom="web", metric="ce"):
        return ev[seed, cond, stage, step, role, dom][metric]

    # ---------- independent recomputation of the contrasts ----------
    rows = {}
    for seed in SEEDS:
        def contrast(step, role="full"):
            rp, tp = ce(seed, "RMS", "prefix", PREFIX_END, role), ce(seed, "Taper-minus", "prefix", PREFIX_END, role)
            rw, rpy = ce(seed, "RMS", "web", step, role), ce(seed, "RMS", "python", step, role)
            tw, tpy = ce(seed, "Taper-minus", "web", step, role), ce(seed, "Taper-minus", "python", step, role)
            d = (tpy - tw) - (rpy - rw)
            g = (tp - rp) - (tw - rw)
            return {"D": d, "G": g, "Q": d - g, "F_rms_web": rw - rp, "F_rms_py": rpy - rp,
                    "F_t_web": tw - tp, "F_t_py": tpy - tp}
        end = contrast(CONT)
        assert abs(end["Q"] - (end["F_t_py"] - end["F_rms_py"])) < 1e-9
        rep = next(s for s in report["seeds"] if s["seed"] == seed)
        for key, theirs in (("D", rep["endpoint"]["d"]), ("G", rep["endpoint"]["g_web"]), ("Q", rep["endpoint"]["q"])):
            assert abs(end[key] - theirs) < 1e-12, (seed, key)
        for u in PERSIST:
            assert abs(contrast(u)["D"] - rep["full_dev_persistence"][str(u)]["d"]) < 1e-12
        prefix = {c: ce(seed, c, "prefix", PREFIX_END) for c in CONDS}
        gap = abs(prefix["Taper-minus"] - prefix["RMS"]) / prefix["RMS"]
        assert abs(gap - rep["evidence"]["absolute_relative_prefix_gap"]) < 1e-12
        adapt = {c: ce(seed, c, "prefix", PREFIX_END, dom="python", metric="non_w_ce")
                 - ce(seed, c, "python", CONT, dom="python", metric="non_w_ce") for c in CONDS}
        rows[seed] = {**end, "D1525": contrast(1525)["D"], "D3050": contrast(3050)["D"], "prefix": prefix,
                      "gap": gap, "adapt": adapt, "matched": rep["matched"], "tail": rep["tail"],
                      "classes": rep["class_contributions"], "slopes": rep["prefix_slopes_nats_per_million_tokens"],
                      "full_D": {u: contrast(u)["D"] for u in FULL_CONT},
                      "quick_D": {u: contrast(u, "quick")["D"] for u in QUICK_CONT}}
    assert report["complete"] and not report["failures"] and report["missing_event_count"] == 0
    assert report["decision"]["category"] == "STOP — SMALL OBSERVED EFFECT"
    thresholds = config["decision_thresholds"]
    assert thresholds == {"prefix_gap": 0.02, "adaptation": 0.05, "proceed_d": 0.03, "proceed_d3050": 0.015,
                          "q_fraction": 0.5, "opposite_d": -0.03, "small_d": 0.015, "transient_d": 0.05}

    Ds = [rows[s]["D"] for s in SEEDS]
    mean_d, sd_d = statistics.mean(Ds), statistics.stdev(Ds)
    half = 4.303 * sd_d / math.sqrt(3)
    assert abs(mean_d - report["uncertainty"]["mean"]) < 1e-12
    early = {u: statistics.mean(rows[s]["quick_D"][u] for s in SEEDS) for u in QUICK_CONT if u <= 1000}
    early_max_u = max(early, key=early.get)
    f_code = [rows[s][k] for s in SEEDS for k in ("F_rms_py", "F_t_py")]
    f_web = [rows[s][k] for s in SEEDS for k in ("F_rms_web", "F_t_web")]
    mean_f_code = statistics.mean(f_code)
    target = report["decision"]["common_target_update"]
    matched_target = [rows[s]["matched"][str(target)]["differential_forgetting"] for s in SEEDS]
    clip = {c: [completion[s, c]["clipped"] / completion[s, c]["updates"] for s in SEEDS] for c in CONDS}
    retries = sum(completion[k]["overflow_retries"] for k in completion)
    sec_per_update = {c: statistics.mean(completion[s, c]["optimizer_seconds"] / completion[s, c]["updates"]
                                         for s in SEEDS) for c in CONDS}
    n_docs_web = len(ev[101, "RMS", "web", CONT, "full", "web"]["doc_sums"])
    audit = json.loads((ONLINE / "audit.json").read_text())
    manifest = json.loads((ONLINE / "manifest.json").read_text())
    start = dt.datetime.fromtimestamp(session["started_unix"], dt.UTC)
    end = dt.datetime.fromtimestamp(session["ended_unix"], dt.UTC)

    # ---------- parameter counts (architecture is fixed by the protocol) ----------
    emb, pos = 50257 * 256, 512 * 256
    block = 256 * 768 + 256 * 256 + 256 * 1024 + 1024 * 256
    params_rms = emb + pos + 6 * (block + 2 * 256) + 256
    params_taper = params_rms + 12 * 256  # gamma-tilde per internal site

    def m(x, digits=4, sign=True):
        return "\\ensuremath{" + (f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}").replace("+", "{+}") + "}"

    macros = {
        "Dmean": m(mean_d), "Dsd": m(sd_d, sign=False), "Dlo": m(mean_d - half), "Dhi": m(mean_d + half),
        "Dmin": m(min(Ds)), "Dmax": m(max(Ds)),
        "DthreeMean": m(statistics.mean(rows[s]["D3050"] for s in SEEDS)),
        "DoneMean": m(statistics.mean(rows[s]["D1525"] for s in SEEDS)),
        "Qmean": m(statistics.mean(rows[s]["Q"] for s in SEEDS)),
        "Gmean": m(statistics.mean(rows[s]["G"] for s in SEEDS)),
        "matchedMean": m(statistics.mean(matched_target)), "matchedTarget": f"{target:,}",
        "FcodeMin": m(min(f_code), 2, sign=False), "FcodeMax": m(max(f_code), 2, sign=False),
        "FcodeMean": m(mean_f_code, 2, sign=False),
        "FwebMin": m(min(f_web), 3), "FwebMax": m(max(f_web), 3),
        "DrelPct": f"{abs(mean_d) / mean_f_code * 100:.1f}",
        "gapMin": f"{min(rows[s]['gap'] for s in SEEDS) * 100:.2f}", "gapMax": f"{max(rows[s]['gap'] for s in SEEDS) * 100:.2f}",
        "adaptMin": f"{min(v for s in SEEDS for v in rows[s]['adapt'].values()):.2f}",
        "adaptMax": f"{max(v for s in SEEDS for v in rows[s]['adapt'].values()):.2f}",
        "earlyMax": m(early[early_max_u], 3), "earlyMaxAt": f"{early_max_u:,}",
        "clipRMSlo": f"{min(clip['RMS']) * 100:.0f}", "clipRMShi": f"{max(clip['RMS']) * 100:.0f}",
        "clipTaperlo": f"{min(clip['Taper-minus']) * 100:.0f}", "clipTaperhi": f"{max(clip['Taper-minus']) * 100:.0f}",
        "overflowRetries": f"{retries}",
        "secRMS": f"{sec_per_update['RMS']:.2f}", "secTaper": f"{sec_per_update['Taper-minus']:.2f}",
        "sessionHours": f"{session['hours']:.2f}",
        "sessionStartTime": start.strftime("%H:%M"), "sessionEndTime": end.strftime("%H:%M"),
        "sessionStart": f"{start.day} {start.strftime('%B %Y')}, {start.strftime('%H:%M')}",
        "sessionEnd": f"{end.day} {end.strftime('%B %Y')}, {end.strftime('%H:%M')}",
        "torchVersion": session["torch"].split("+")[0], "cudaVersion": session["cuda"],
        "paramsRMS": f"{params_rms:,}", "paramsTaper": f"{params_taper:,}",
        "embedShare": f"{emb / params_rms * 100:.0f}",
        "nDocsWeb": f"{n_docs_web:,}",
        "medianDocMin": m(min(rows[s]["tail"]["unweighted_median"] for s in SEEDS), 3),
        "medianDocMax": m(max(rows[s]["tail"]["unweighted_median"] for s in SEEDS), 3),
        "configSha": config["config_sha256"][:16], "runnerSha": session["code_sha256"][:16],
        "nearRemoved": f"{audit['near_removed']:,}", "exactRemoved": f"{audit['exact_removed']:,}",
        "repoGroups": f"{audit['repository_groups']:,}", "auditPairs": f"{audit['missed_candidate_pairs_checked']:,}",
        "auditFound": f"{audit['missed_candidate_pairs_found']}",
        "webTrainTokens": f"{manifest['splits']['web_train']['tokens']:,}",
        "pyTrainTokens": f"{manifest['splits']['python_train']['tokens']:,}",
        "cfourRev": config["data"]["sources"]["web"]["revision"][:12],
        "stackRev": config["data"]["sources"]["python"]["revision"][:12],
        "slopeRMS": m(statistics.mean(rows[s]["slopes"]["RMS"] for s in SEEDS), 5),
        "slopeTaper": m(statistics.mean(rows[s]["slopes"]["Taper-minus"] for s in SEEDS), 5),
    }
    traj = {u: statistics.mean(rows[s]["full_D"][u] for s in SEEDS) for u in FULL_CONT if u > 0}
    lo_u, hi_u = min(traj, key=traj.get), max(traj, key=traj.get)
    seed_abs = max(abs(rows[s]["full_D"][u]) for s in SEEDS for u in FULL_CONT)
    late = [u for u in FULL_CONT if u >= 305]
    max_jump = max(abs(rows[s]["full_D"][b] - rows[s]["full_D"][a]) for s in SEEDS for a, b in zip(late, late[1:]))
    macros["maxJump"] = m(max_jump, 3, sign=False)
    shares = {s: {c["token_class"]: c["weighted_contribution"] for c in rows[s]["classes"]} for s in SEEDS}
    a_negative = sum(shares[s]["A"] < 0 for s in SEEDS)
    a_largest = sum(min(shares[s], key=shares[s].get) == "A" for s in SEEDS)
    macros["classAneg"], macros["classAlargest"] = f"{a_negative}", f"{a_largest}"
    pairs = [(s, u) for s in SEEDS for u in PERSIST if rows[s]["matched"][str(u)]["match"] is not None]
    earlier = sum(rows[s]["matched"][str(u)]["match"]["update"] < u for s, u in pairs)
    assert len(pairs) == 9  # every seed reached every target
    macros["matchedEarlier"], macros["matchedPairs"] = f"{earlier}", f"{len(pairs)}"
    for seed in SEEDS:
        assert abs(rows[seed]["full_D"][0]) < 1e-6  # both branches start from the identical switch state
    macros.update({"trajMin": m(traj[lo_u], 3), "trajMinAt": f"{lo_u:,}", "trajMax": m(traj[hi_u], 3),
                   "trajMaxAt": f"{hi_u:,}", "seedAbsMax": m(seed_abs, 3, sign=False),
                   "nNegEnd": f"{sum(rows[s]['D'] < 0 for s in SEEDS)}"})
    for seed, letter in zip(SEEDS, "ABC"):
        macros[f"D{letter}"] = m(rows[seed]["D"])
        macros[f"DoneFive{letter}"] = m(rows[seed]["D1525"], 3)
    macros.update(second_draft_analyses(ev, rows, config, session))

    # ---- abstract: one ASCII text for both the PDF and the arXiv metadata field
    seeds_txt = ", ".join(f"{rows[s]['D']:.3f}" for s in SEEDS)
    abstract = (
        "TaperNorm replaces a pre-norm Transformer's internal normalization with a gated map that acts like "
        "RMSNorm early in training and then becomes a fixed, calibrated linear scaling. Without per-token "
        "normalization, such a model might forget more after a shift in the training data. We tested this in a "
        "small pilot with design, endpoint and decision rules fixed before training. Paired "
        "17.7M-parameter models with internal RMSNorm or TaperNorm (three seeds) were trained on 150M web "
        "tokens, then continued for 100M tokens on web text or Python code. The primary endpoint is a "
        "difference-in-differences D in held-out web "
        "cross-entropy; positive D means extra forgetting under TaperNorm. Switching to Python raised web "
        f"cross-entropy by about {mean_f_code:.2f} nats/token in both models. Mean D was {mean_d:.3f} nats/token "
        f"(seeds {seeds_txt}), below the pre-specified +0.015 bound, so the pre-specified decision is to stop: "
        "we find no evidence that TaperNorm increases persistent forgetting in this setting. In exploratory "
        "analyses, D was near zero for most of continuation (after a brief early dip) and became negative as "
        "the learning rate annealed, and the code-specific activation-scale shift that motivated the hypothesis "
        "was small in both models. Code and all evaluation records are released.")
    assert abstract.isascii(), "abstract must be plain ASCII"
    (ROOT / "paper" / "arxiv-abstract.txt").write_text(abstract + "\n", encoding="ascii")
    (OUT / "abstract.tex").write_text(abstract + "\n", encoding="utf-8")
    print(f"abstract: {len(abstract)} characters, ASCII")
    lines = ["% Generated by paper/make_assets.py from reports/h1-kaggle -- do not edit by hand."]
    # Thousands separators as {,} so numbers typeset correctly in both text and math mode.
    lines += [f"\\newcommand{{\\{k}}}{{{v.replace(',', '{,}')}}}" for k, v in macros.items()]
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def fmt(x, d=4):
        return f"${x:+.{d}f}$".replace("+", "{+}")

    # ---------- Table: primary contrasts per seed ----------
    t = ["\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule",
         "Seed & $D$ & $D(3050)$ & $D(1525)$ & $G_{\\text{web}}$ & $Q$ & Matched ($s{=}" + f"{target}" + "$) \\\\",
         "\\midrule"]
    for seed in SEEDS:
        r = rows[seed]
        t.append(f"{seed} & {fmt(r['D'])} & {fmt(r['D3050'])} & {fmt(r['D1525'])} & {fmt(r['G'])} & {fmt(r['Q'])} & "
                 f"{fmt(r['matched'][str(target)]['differential_forgetting'])} \\\\")
    t.append("\\midrule")
    means = {k: statistics.mean(rows[s][k] for s in SEEDS) for k in ("D", "D3050", "D1525", "G", "Q")}
    sds = {k: statistics.stdev(rows[s][k] for s in SEEDS) for k in ("D", "D3050", "D1525", "G", "Q")}
    t.append("Mean & " + " & ".join(fmt(means[k]) for k in ("D", "D3050", "D1525", "G", "Q")) + f" & {fmt(statistics.mean(matched_target))} \\\\")
    t.append("SD & " + " & ".join(f"${sds[k]:.4f}$" for k in ("D", "D3050", "D1525", "G", "Q"))
             + f" & ${statistics.stdev(matched_target):.4f}$ \\\\")
    t += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "table_primary.tex").write_text("\n".join(t) + "\n", encoding="utf-8")

    # ---------- Table: validity guardrails and actual forgetting ----------
    t = ["\\begin{tabular}{@{}lrrrrrrrr@{}}", "\\toprule",
         " & \\multicolumn{2}{c}{Prefix web CE} & & \\multicolumn{2}{c}{$F$, web branch} & \\multicolumn{2}{c}{$F$, Python branch} & \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
         "Seed & RMS & Taper & Gap & RMS & Taper & RMS & Taper & Code gain (R / T) \\\\", "\\midrule"]
    for seed in SEEDS:
        r = rows[seed]
        t.append(f"{seed} & ${r['prefix']['RMS']:.4f}$ & ${r['prefix']['Taper-minus']:.4f}$ & ${r['gap'] * 100:.2f}\\%$ & "
                 f"{fmt(r['F_rms_web'], 3)} & {fmt(r['F_t_web'], 3)} & {fmt(r['F_rms_py'], 3)} & {fmt(r['F_t_py'], 3)} & "
                 f"${r['adapt']['RMS']:.2f}$ / ${r['adapt']['Taper-minus']:.2f}$ \\\\")
    t += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "table_validity.tex").write_text("\n".join(t) + "\n", encoding="utf-8")

    # ---------- Table: matched adaptation (appendix) ----------
    t = ["\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule",
         " & \\multicolumn{3}{c}{Matched differential forgetting} & \\multicolumn{3}{c}{Taper update at match (bracket)} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
         "Seed & $s{=}1525$ & $s{=}3050$ & $s{=}6104$ & $s{=}1525$ & $s{=}3050$ & $s{=}6104$ \\\\", "\\midrule"]
    for seed in SEEDS:
        cells, brackets = [], []
        for u in PERSIST:
            item = rows[seed]["matched"][str(u)]
            cells.append(fmt(item["differential_forgetting"]) if item["differential_forgetting"] is not None else "--")
            mt = item["match"]
            brackets.append(f"{mt['update']:.0f} ({mt['left']['update']}--{mt['right']['update']})" if mt else "--")
        t.append(f"{seed} & " + " & ".join(cells) + " & " + " & ".join(brackets) + " \\\\")
    t += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "table_matched.tex").write_text("\n".join(t) + "\n", encoding="utf-8")

    # ---------- Table: token-class decomposition and document tails (appendix) ----------
    t = ["\\begin{tabular}{@{}lrrrrrrr@{}}", "\\toprule",
         " & \\multicolumn{4}{c}{Contribution to $D$ by token class} & \\multicolumn{3}{c}{Per-document $D_i$} \\\\",
         "\\cmidrule(lr){2-5}\\cmidrule(lr){6-8}",
         "Seed & W & A & P & X & Median & 1\\%-trimmed & Top-1\\% sum \\\\", "\\midrule"]
    for seed in SEEDS:
        cls = {c["token_class"]: c for c in rows[seed]["classes"]}
        tail = rows[seed]["tail"]
        t.append(f"{seed} & " + " & ".join(fmt(cls[c]["weighted_contribution"]) for c in "WAPX") + " & "
                 + f"{fmt(tail['unweighted_median'])} & {fmt(tail['trimmed_token_weighted_mean'])} & {fmt(tail['top_signed_sum'])} \\\\")
    counts = {c["token_class"]: c["count"] for c in rows[101]["classes"]}
    t.append("Labels & " + " & ".join(f"{counts[c]:,}" for c in "WAPX") + " & \\multicolumn{3}{c}{" + f"{n_docs_web:,} documents" + "} \\\\")
    t += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "table_tails.tex").write_text("\n".join(t) + "\n", encoding="utf-8")
    for seed in SEEDS:
        cls_sum = sum(c["weighted_contribution"] for c in rows[seed]["classes"])
        assert abs(cls_sum - rows[seed]["D"]) < 1e-6

    # ---------- Figure: trajectories ----------
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5,
                         "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.labelcolor": INK2, "text.color": INK, "legend.fontsize": 6.3,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.35), constrained_layout=True)
    xs = FULL_CONT

    def style(ax, title):
        ax.set_title(title, loc="left", color=INK, fontweight="bold")
        ax.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlim(-150, 6300)
        ax.set_xticks([0, 1525, 3050, 4575, 6104])
        ax.set_xlabel("Continuation update $s$")

    # (a) web CE in all four arms: color = condition, line style = branch (also labeled directly)
    ax = axes[0]
    for cond in CONDS:
        for branch, ls in (("python", "-"), ("web", (0, (3, 1.6)))):
            ys = [statistics.mean(ce(s, cond, branch, u) for s in SEEDS) for u in xs]
            ax.plot(xs, ys, color=COLOR[cond], linestyle=ls, linewidth=1.4, solid_capstyle="round",
                    label=cond if branch == "python" else None)
    style(ax, "(a) Held-out web CE")
    ax.set_ylabel("Full-dev web CE (nats/token)")
    ax.text(6104, 6.32, "Python branches (solid)", color=INK2, fontsize=6.2, ha="right")
    ax.text(6104, 4.56, "web branches (dashed)", color=INK2, fontsize=6.2, ha="right")
    ax.set_ylim(4.35, 6.45)
    ax.legend(loc="center right", frameon=False, handlelength=1.8)

    # (b) D(s) per seed and mean; pre-registered thresholds are the labeled y-ticks
    ax = axes[1]
    ax.axvspan((4880 + 5185) / 2, 6300, color="#f0efec", zorder=0, linewidth=0)  # post hoc late window
    for y in (0.03, 0.015, -0.03):
        ax.axhline(y, color=MUTED, linewidth=0.6)
    ax.axhline(0, color=INK2, linewidth=0.7)
    for seed, marker in zip(SEEDS, ("o", "s", "^")):
        ax.plot(xs, [rows[seed]["full_D"][u] for u in xs], color=MUTED, linewidth=0.8, marker=marker,
                markersize=2.6, markeredgewidth=0, label=f"seed {seed}")
    ax.plot(xs, [statistics.mean(rows[s]["full_D"][u] for s in SEEDS) for u in xs], color=INK, linewidth=1.6,
            label="mean of seeds")
    style(ax, "(b) Contrast $D(s)$, full dev")
    ax.grid(False, axis="y")
    ax.set_yticks([-0.06, -0.03, 0, 0.015, 0.03, 0.06])
    ax.set_yticklabels(["−0.06", "−0.03", "0", "+0.015", "+0.03", "+0.06"])
    ax.set_ylabel("$D(s)$ (nats/token)")
    ax.set_ylim(-0.068, 0.068)
    ax.legend(loc="lower left", frameon=False, ncol=2, columnspacing=0.8, handlelength=1.6)

    # (c) code adaptation in the Python branch
    ax = axes[2]
    for cond in CONDS:
        ys = [statistics.mean(ce(s, cond, "python", u, dom="python", metric="non_w_ce") for s in SEEDS) for u in xs]
        ax.plot(xs, ys, color=COLOR[cond], linewidth=1.4, label=cond)
    style(ax, "(c) Code adaptation")
    ax.set_ylabel("Full-dev non-W code CE (nats/token)")
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(FIG / "trajectories.pdf", metadata={"CreationDate": None})  # no timestamp: byte-reproducible
    fig.savefig(FIG / "trajectories.png", dpi=220)
    plt.close(fig)

    summary = {k: v for k, v in macros.items()}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("independent recomputation agrees with the frozen report; wrote", len(macros), "macros, 4 tables, 1 figure")
    for k in ("Dmean", "Dsd", "Dlo", "Dhi", "DthreeMean", "Qmean", "Gmean", "matchedMean", "FcodeMean", "DrelPct",
              "earlyMax", "earlyMaxAt", "clipRMSlo", "clipRMShi", "clipTaperlo", "clipTaperhi", "paramsRMS",
              "embedShare", "nDocsWeb", "slopeRMS", "slopeTaper", "sessionStart", "sessionEnd"):
        print(f"  {k} = {macros[k]}")


if __name__ == "__main__":
    main()
