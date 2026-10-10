"""Generate every number, table and figure in the paper from the run's own records.

Reads reports/h1-kaggle/ (Study 1: decision report, config, session record, per-run evaluation logs,
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
    out["earlyLo"] = mm(min(point_mean[u] for u in early), 4)  # 4 decimals, as \dipMean (same s = 100 value)
    out["earlyHi"] = mm(max(point_mean[u] for u in early), 4)
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


S2DIR = ROOT / "reports" / "s2-kaggle"
S2STATE = S2DIR / "run" / "s2state"
SEEDS2 = (101, 102, 103, 104, 105, 106)
FRESH = (104, 105, 106)
T95 = {3: 4.303, 6: 2.571}   # pre-registered (study2/PROTOCOL.md, Sec. 8.5)
T90 = {3: 2.920, 6: 2.015}


def figure_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5,
                         "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.labelcolor": INK2, "text.color": INK, "legend.fontsize": 6.3,
                         "pdf.fonttype": 42})


def sha256_lf(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_study2():
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    data = {"report": read(S2STATE / "decision-report.json"), "config": read(S2STATE / "config.json"),
            "sessions": [json.loads(l) for l in (S2STATE / "sessions.jsonl").read_text().splitlines() if l.strip()],
            "selection": read(S2DIR / "probe" / "s2probe" / "selection.json"),
            "online": read(S2DIR / "prepare" / "s2prep" / "s2online" / "manifest.json"),
            "audit": read(S2DIR / "prepare" / "s2prep" / "s2online" / "audit.json"),
            "acceptance": read(S2DIR / "prepare" / "s2prep" / "acceptance.json"),
            "smoke": read(S2DIR / "smoke" / "smoke-results.json"),
            "stamp": read(ROOT / "study2" / "timestamp-evidence.json"),
            "registration": read(ROOT / "study2" / "registration-manifest.json")}
    ev2, completion2 = {}, {}
    for seed in SEEDS2:
        for cond in CONDS:
            run = S2STATE / "runs" / f"S{seed}-{cond}"
            completion2[seed, cond] = read(run / "completion.json")
            for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                ev2[seed, cond, r["stage"], r["step"], r["role"], r["domain"]] = r
    data["ev"], data["completion"] = ev2, completion2
    return data


def interval(values, table):
    mean, sd = statistics.mean(values), statistics.stdev(values)
    half = table[len(values)] * sd / math.sqrt(len(values))
    return mean, sd, mean - half, mean + half


def study2_assets(ev1, m, fmt):
    """Study 2: integrity chain, independent recomputation of every H2/R2 contrast and of the manipulation check
    against the frozen report, then macros, tables and the Study 2 figure."""
    d = load_study2()
    rep, cfg, sel, ev2 = d["report"], d["config"], d["selection"], d["ev"]
    out = {}
    # ---- integrity chain: configuration, registration manifest, runner and protocol hashes
    body = {k: v for k, v in cfg.items() if k not in ("config_sha256", "checkpoint_seconds")}
    assert hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest() \
        == cfg["config_sha256"], "Study 2 config hash does not recompute"
    manifest_path = ROOT / "study2" / "registration-manifest.json"
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert manifest_sha == d["stamp"]["manifest_sha256"] == sel["registration_manifest_sha256"] \
        == d["online"]["registration_manifest_sha256"]
    registered = d["registration"]["files_sha256"]
    for session in d["sessions"]:
        assert session["registration_manifest_sha256"] == manifest_sha
        assert session["code_sha256"] == registered["kaggle_s2/s2_run.py"]
    assert sha256_lf(ROOT / "kaggle_s2" / "s2_run.py") == registered["kaggle_s2/s2_run.py"]
    assert sha256_lf(ROOT / "study2" / "PROTOCOL.md") == registered["study2/PROTOCOL.md"], \
        "the registered protocol must stay byte-identical"
    try:  # the OpenTimestamps proof, if the library is installed
        from opentimestamps.core.notary import BitcoinBlockHeaderAttestation
        from opentimestamps.core.serialize import StreamDeserializationContext
        from opentimestamps.core.timestamp import DetachedTimestampFile
        with (ROOT / "study2" / "registration-manifest.json.ots").open("rb") as stream:
            proof = DetachedTimestampFile.deserialize(StreamDeserializationContext(stream))
        assert proof.file_digest.hex() == manifest_sha
        heights = {a.height for _, a in proof.timestamp.all_attestations()
                   if isinstance(a, BitcoinBlockHeaderAttestation)}
        assert d["stamp"]["bitcoin_block_height"] in heights
    except ImportError:
        print("note: opentimestamps not installed; the .ots proof was not re-parsed")
    assert rep["complete"] and not rep["failures"] and not rep["problems"] and rep["missing_event_count"] == 0
    assert rep["h2"]["decision"]["category"] == "OPPOSITE DIRECTION"
    assert rep["r2"]["decision"]["category"] == "STOP — SMALL OBSERVED EFFECT"
    assert cfg["decision_thresholds"] == {"prefix_gap": 0.02, "adaptation": 0.05, "proceed_d": 0.03,
                                          "proceed_d3050": 0.015, "q_fraction": 0.5, "opposite_d": -0.03,
                                          "small_d": 0.015, "transient_d": 0.05}
    lineage = rep["lineage"]
    assert len(lineage) == 24 and all(v["passed"] and v["max_ce_diff"] == 0.0 and v["max_energy_rel_diff"] == 0.0
                                      for v in lineage.values())
    for seed in SEEDS:  # the reused switch states re-evaluate bit-exactly to the pilot's own records
        for cond in CONDS:
            for role in ("full", "quick"):
                assert ev2[seed, cond, "x", 0, role, "web"]["ce"] == ev1[seed, cond, "prefix", PREFIX_END, role, "web"]["ce"]
    smoke = d["smoke"]
    assert smoke["lineage"]["passed"] and all(r["max_ce_diff"] == 0.0 for r in smoke["lineage"]["runs"].values())
    assert sel["status"] == "selected" and sel["selected"] == "mc4-zh" and sel["reproduction"]["passed"]

    # ---- independent recomputation of the contrasts
    def ce2(seed, cond, stage, step, role="full", dom="web", metric="ce"):
        src = ev1 if seed in SEEDS and stage != "x" else ev2
        return src[seed, cond, stage, step, role, dom][metric]

    def contrast2(seed, shift, step, role="full"):
        rp, tp = ce2(seed, "RMS", "prefix", PREFIX_END, role), ce2(seed, "Taper-minus", "prefix", PREFIX_END, role)
        rw, rs = ce2(seed, "RMS", "web", step, role), ce2(seed, "RMS", shift, step, role)
        tw, ts = ce2(seed, "Taper-minus", "web", step, role), ce2(seed, "Taper-minus", shift, step, role)
        dd, g = (ts - tw) - (rs - rw), (tp - rp) - (tw - rw)
        return {"D": dd, "G": g, "Q": dd - g, "F_rms_web": rw - rp, "F_rms": rs - rp, "F_t_web": tw - tp, "F_t": ts - tp}

    rows = {}
    for shift, key, seeds in (("x", "h2", SEEDS2), ("python", "r2", SEEDS2)):
        details = {s["seed"]: s for s in rep[key]["seeds"]} if key == "h2" else \
            {s["seed"]: s for s in rep["r2"]["seeds"]}
        for seed in seeds:
            end = contrast2(seed, shift, CONT)
            base = ("prefix", PREFIX_END) if shift == "python" else ("x", 0)
            adapt = {c: ce2(seed, c, *base, dom=shift, metric="non_w_ce") - ce2(seed, c, shift, CONT, dom=shift,
                                                                                metric="non_w_ce") for c in CONDS}
            prefix = {c: ce2(seed, c, "prefix", PREFIX_END) for c in CONDS}
            row = {**end, "D3050": contrast2(seed, shift, 3050)["D"], "D1525": contrast2(seed, shift, 1525)["D"],
                   "gap": abs(prefix["Taper-minus"] - prefix["RMS"]) / prefix["RMS"], "adapt": adapt,
                   "full_D": {u: contrast2(seed, shift, u)["D"] for u in FULL_CONT},
                   "quick_D": {u: contrast2(seed, shift, u, "quick")["D"] for u in QUICK_CONT}}
            if seed in details:
                found = details[seed]
                assert abs(row["D"] - found["endpoint"]["d"]) < 1e-12 and abs(row["Q"] - found["endpoint"]["q"]) < 1e-12
                assert abs(row["G"] - found["endpoint"]["g_web"]) < 1e-12
                assert abs(row["D3050"] - found["evidence"]["d3050"]) < 1e-12
                row.update(matched=found["matched"], tail=found["tail"], classes=found["class_contributions"],
                           rare=found["rare"])
            rows[shift, seed] = row
    for seed in SEEDS:  # the pilot's own web -> Python values, recomputed through the Study 2 path
        assert abs(rows["python", seed]["D"] - contrast2(seed, "python", CONT)["D"]) < 1e-15
    h2 = [rows["x", s]["D"] for s in SEEDS2]
    r2 = [rows["python", s]["D"] for s in FRESH]
    pooled = [rows["python", s]["D"] for s in SEEDS2]
    for values, block in ((h2, rep["h2"]["uncertainty"]), (r2, rep["r2"]["uncertainty"]),
                          (pooled, rep["pooled_python_descriptive"]),
                          ([rows["x", s]["D"] for s in FRESH], rep["h2"]["fresh_seed_sensitivity"])):
        mean, sd, lo, hi = interval(values, T95)
        _, _, lo90, hi90 = interval(values, T90)
        assert abs(mean - block["mean"]) < 1e-12 and abs(lo - block["t95"][0]) < 1e-12 and abs(hi90 - block["t90"][1]) < 1e-12

    # ---- manipulation check (Finding A's measures), recomputed from the diagnostic probes
    def diag2(seed, cond, stage, step):
        if seed in SEEDS and stage != "x":
            return ev1[seed, cond, stage, step, "diag", "both"]["measurement"]
        return ev2[seed, cond, stage, step, "diag", "all"]["measurement"]

    sites = [f"{i}.{b}" for i in range(6) for b in ("attention", "mlp")]
    scale = {}
    for shift in ("x", "python"):
        for cond in CONDS:
            gap, wch, sch, spec = [], [], [], []
            for seed in SEEDS2:
                switch = diag2(seed, cond, "prefix", PREFIX_END)
                gsrc = diag2(seed, cond, "x", 0) if shift == "x" else switch
                wend, send = diag2(seed, cond, "web", CONT), diag2(seed, cond, shift, CONT)
                for site in sites:
                    E = lambda mm_, dom, site=site: mm_["sites"][f"{site}.h"]["all"][dom]["mean_squared_norm"]
                    gap.append(0.5 * math.log(E(gsrc, shift) / E(gsrc, "web")))
                    wch.append(0.5 * math.log(E(wend, "web") / E(switch, "web")))
                    sch.append(0.5 * math.log(E(send, "web") / E(switch, "web")))
                    spec.append(sch[-1] - wch[-1])
            v = {"gap": float(np.mean(np.abs(gap))), "w_abs": float(np.mean(np.abs(wch))),
                 "s_abs": float(np.mean(np.abs(sch))), "w_signed": float(np.mean(wch)),
                 "s_signed": float(np.mean(sch)), "specific": float(np.mean(np.abs(spec)))}
            frozen = rep["manipulation_check"][shift][cond]
            for mine, theirs in (("gap", "switch_gap_abs"), ("w_abs", "web_branch_change_abs"),
                                 ("s_abs", "shift_branch_change_abs"), ("specific", "specific_change_abs")):
                assert abs(v[mine] - frozen[theirs]) < 1e-12, (shift, cond, mine)
            scale[shift, cond] = v
    reading = rep["manipulation_check"]["reading"]
    assert reading == {"switch_gap_larger_for_x": False, "specific_change_larger_for_x": True}

    # ---- macros
    cand = sel["candidates"]
    out.update({"twoSelZh": f"{cand['mc4-zh']['S']:.4f}", "twoSelRu": f"{cand['mc4-ru']['S']:.4f}",
                "twoSelDe": f"{cand['mc4-de']['S']:.4f}", "twoSelOwm": f"{cand['openwebmath']['S']:.4f}",
                "twoSelPy": f"{sel['python_reference']['S']:.4f}",
                "twoSelPyR": f"{sel['python_reference']['per_condition']['RMS']:.4f}",
                "twoSelPyT": f"{sel['python_reference']['per_condition']['Taper-minus']:.4f}",
                "twoReproMax": f"{max(sel['reproduction']['max_relative_difference'].values()):.1f}"})
    online, audit, acc = d["online"], d["audit"], d["acceptance"]
    out.update({"twoTrainTokens": f"{online['splits']['x_train']['tokens']:,}",
                "twoTrainDocs": f"{acc['documents']['x_train']:,}", "twoDevDocs": f"{acc['documents']['x_dev']:,}",
                "twoTestDocs": f"{acc['documents']['x_test']:,}",
                "twoNearRemoved": f"{audit['near_removed']:,}", "twoExactRemoved": f"{audit['exact_removed']:,}",
                "twoFrozenRemoved": f"{audit['exact_frozen_removed'] + audit['near_frozen_dev_removed']:,}",
                "twoAuditPairs": f"{audit['missed_candidate_pairs_checked']:,}",
                "twoFrozenPairs": f"{audit['frozen_pairs_checked']:,}",
                "twoTrainShards": f"{len(online['domain']['shards_consumed']['train'])}",
                "twoValShards": f"{len(online['domain']['shards_consumed']['validation'])}",
                "twoRawTrain": f"{online['raw_tokens_before_dedup']['train'] / 1e6:.1f}"})
    sessions = d["sessions"]
    stamp = d["stamp"]
    to_dt = lambda s: dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    first_job = to_dt(stamp["kaggle_versions_embedding_the_manifest_utc"]["danny00/s2-domain-probe v1"])
    block_time = to_dt(stamp["bitcoin_block_time_utc"])
    submit = to_dt(stamp["calendar_submission_utc"])
    run_start = dt.datetime.fromtimestamp(sessions[0]["started_unix"], dt.UTC)
    when = lambda t: f"{t.strftime('%H:%M')} UTC on {t.day} {t.strftime('%B')}"
    sec2 = {c: statistics.mean(d["completion"][s, c]["optimizer_seconds"] / d["completion"][s, c]["updates"]
                               for s in FRESH) for c in CONDS}
    clip2 = rep["clipping"]
    out.update({"twoSessOne": f"{sessions[0]['hours']:.2f}", "twoSessTwo": f"{sessions[1]['hours']:.2f}",
                "twoSessTotal": f"{sum(s['hours'] for s in sessions):.2f}",
                "twoTorch": sessions[0]["torch"].split("+")[0],
                "twoManifestSha": manifest_sha[:12], "twoSubmit": when(submit), "twoFirstJob": when(first_job),
                "twoRunStart": when(run_start), "twoBlock": f"{stamp['bitcoin_block_height']:,}",
                "twoBlockTime": when(block_time),
                "twoBlockAfterStart": f"{(block_time - run_start).total_seconds() / 3600:.0f}",
                "twoLineageN": f"{len(lineage)}", "twoSecRMS": f"{sec2['RMS']:.2f}", "twoSecTaper": f"{sec2['Taper-minus']:.2f}",
                "twoClipXR": f"{clip2['RMS']['x']['mean'] * 100:.1f}",
                "twoClipXT": f"{clip2['Taper-minus']['x']['mean'] * 100:.1f}",
                "twoClipPyR": f"{clip2['RMS']['python']['mean'] * 100:.0f}",
                "twoClipWebR": f"{clip2['RMS']['web']['mean'] * 100:.0f}"})
    mean, sd, lo, hi = interval(h2, T95)
    _, _, lo90, hi90 = interval(h2, T90)
    fmean, _, flo, fhi = interval([rows["x", s]["D"] for s in FRESH], T95)
    f_x = [rows["x", s][k] for s in SEEDS2 for k in ("F_rms", "F_t")]
    early = {u: statistics.mean(rows["x", s]["quick_D"][u] for s in SEEDS2) for u in QUICK_CONT if u <= 1000}
    early_u = max(early, key=early.get)
    traj = {u: statistics.mean(rows["x", s]["full_D"][u] for s in SEEDS2) for u in FULL_CONT if u > 0}
    positive = [s for s in SEEDS2 if rows["x", s]["D"] > 0]
    target = rep["h2"]["decision"]["common_target_update"]
    matched = [rows["x", s]["matched"][str(target)]["differential_forgetting"] for s in SEEDS2]
    boot = rep["document_bootstrap"]
    out.update({"twoDmean": m(mean), "twoDsd": m(sd, sign=False), "twoDlo": m(lo), "twoDhi": m(hi),
                "twoDloNinety": m(lo90), "twoDhiNinety": m(hi90), "twoDmin": m(min(h2)), "twoDmax": m(max(h2)),
                "twoNneg": f"{sum(x < 0 for x in h2)}", "twoPosSeed": f"{positive[0]}" if len(positive) == 1 else "--",
                "twoPosD": m(rows["x", positive[0]]["D"]) if len(positive) == 1 else "--",
                "twoDthree": m(statistics.mean(rows["x", s]["D3050"] for s in SEEDS2)),
                "twoQmean": m(statistics.mean(rows["x", s]["Q"] for s in SEEDS2)),
                "twoGmean": m(statistics.mean(rows["x", s]["G"] for s in SEEDS2)),
                "twoMatched": m(statistics.mean(matched)), "twoMatchedTarget": f"{target:,}",
                "twoGapMin": f"{min(rows['x', s]['gap'] for s in SEEDS2) * 100:.2f}",
                "twoGapMax": f"{max(rows['x', s]['gap'] for s in SEEDS2) * 100:.2f}",
                "twoAdaptMin": f"{min(v for s in SEEDS2 for v in rows['x', s]['adapt'].values()):.2f}",
                "twoAdaptMax": f"{max(v for s in SEEDS2 for v in rows['x', s]['adapt'].values()):.2f}",
                "twoFxMin": m(min(f_x), 2, sign=False), "twoFxMax": m(max(f_x), 2, sign=False),
                "twoFxMean": m(statistics.mean(f_x), 2, sign=False),
                "twoDrelPct": f"{abs(mean) / statistics.mean(f_x) * 100:.1f}",
                "twoFreshMean": m(fmean), "twoFreshLo": m(flo), "twoFreshHi": m(fhi),
                "twoBootLo": m(boot["mean_d_percentile_95"][0]), "twoBootHi": m(boot["mean_d_percentile_95"][1]),
                "twoEarlyMax": m(early[early_u], 3), "twoEarlyMaxAt": f"{early_u:,}",
                "twoTrajNeg": f"{sum(v < 0 for v in traj.values())}", "twoTrajN": f"{len(traj)}",
                "twoMedianMin": m(min(rows["x", s]["tail"]["unweighted_median"] for s in SEEDS2), 3),
                "twoMedianMax": m(max(rows["x", s]["tail"]["unweighted_median"] for s in SEEDS2), 3),
                "twoMedianNeg": f"{sum(rows['x', s]['tail']['unweighted_median'] < 0 for s in SEEDS2)}"})
    cls2 = {s: {c["token_class"]: c["weighted_contribution"] for c in rows["x", s]["classes"]} for s in SEEDS2}
    _, pilot_sd, pilot_lo, pilot_hi = interval([rows["python", s]["D"] for s in SEEDS], T95)
    late = [u for u in FULL_CONT if u >= 5185]  # Study 1's post hoc late window, reused descriptively
    steps = [u for u in FULL_CONT if u >= 305]
    out.update({"twoLateMean": m(statistics.mean(traj[u] for u in late)), "twoLateNeg": f"{sum(traj[u] < 0 for u in late)}",
                "twoLateCount": f"{len(late)}",
                "twoJitter": m(statistics.mean(abs(rows["x", s]["full_D"][b] - rows["x", s]["full_D"][a])
                                               for s in SEEDS2 for a, b in zip(steps, steps[1:])), 3, sign=False)})
    x_classes = ev2[FRESH[0], "RMS", "x", 0, "full", "x"]["classes"]
    w_classes = ev2[FRESH[0], "RMS", "x", 0, "full", "web"]["classes"]
    out["twoCtrlShare"] = f"{x_classes['X'][1] / sum(v[1] for v in x_classes.values()) * 100:.0f}"
    out["webCtrlShare"] = f"{w_classes['X'][1] / sum(v[1] for v in w_classes.values()) * 100:.1f}"
    out.update({"twoPrefixCE":f"{statistics.mean(ce2(s, c, 'prefix', PREFIX_END) for s in SEEDS2 for c in CONDS):.2f}",
                "twoWebEndCE": f"{statistics.mean(ce2(s, c, 'web', CONT) for s in SEEDS2 for c in CONDS):.2f}",
                "twoXendCE": f"{statistics.mean(ce2(s, c, 'x', CONT) for s in SEEDS2 for c in CONDS):.2f}",
                "twoOverflow": f"{sum(d['completion'][s, c]['overflow_retries'] for s in SEEDS2 for c in CONDS)}",
                "twoClassPneg": f"{sum(cls2[s]['P'] < 0 for s in SEEDS2)}",
                "twoClassAneg": f"{sum(cls2[s]['A'] < 0 for s in SEEDS2)}",
                "twoRareMax": f"{max(abs(rows['x', s]['rare']['contribution_to_d']) for s in SEEDS2):.4f}",
                "twoSdRatio": f"{sd / pilot_sd:.0f}",
                "repInPilot": f"{sum(pilot_lo <= rows['python', s]['D'] <= pilot_hi for s in FRESH)}"})
    # post hoc leave-one-seed-out: the category label may change, the permitted reading may not
    loo = {s: statistics.mean(rows["x", u]["D"] for u in SEEDS2 if u != s) for s in SEEDS2}
    flips = [s for s in SEEDS2 if loo[s] > cfg["decision_thresholds"]["opposite_d"]]
    assert all(v < cfg["decision_thresholds"]["small_d"] for v in loo.values()), \
        "every leave-one-out mean must stay in the stop / opposite / transient row of the interpretation matrix"
    out.update({"twoLooMin": m(min(loo.values()), 3), "twoLooMax": m(max(loo.values()), 3),
                "twoLooFlip": " and ".join(str(s) for s in flips), "twoLooFlipMax": m(max(loo[s] for s in flips), 3)})
    # post hoc: R2's sign split coincides with the session split; compare the two seed groups
    split = {}
    for name, seeds in (("pilot", SEEDS), ("fresh", FRESH)):
        g = {}
        for cond, tag in (("RMS", "R"), ("Taper-minus", "T")):
            g[f"prefix{tag}"] = [ce2(s, cond, "prefix", PREFIX_END) for s in seeds]
            g[f"fweb{tag}"] = [rows["python", s]["F_rms_web" if tag == "R" else "F_t_web"] for s in seeds]
            g[f"fpy{tag}"] = [rows["python", s]["F_rms" if tag == "R" else "F_t"] for s in seeds]
            g[f"adapt{tag}"] = [rows["python", s]["adapt"][cond] for s in seeds]
        split[name] = g
    assert all(rows["python", s]["D"] < 0 for s in SEEDS) and all(x > 0 for x in r2), "the sign split in the text"
    for key in ("prefix", "fweb", "fpy"):  # the text says the two groups overlap on each measure
        lo_hi = {n: (min(split[n][key + "R"] + split[n][key + "T"]), max(split[n][key + "R"] + split[n][key + "T"]))
                 for n in split}
        assert lo_hi["pilot"][0] <= lo_hi["fresh"][1] and lo_hi["fresh"][0] <= lo_hi["pilot"][1], key
        for n, (lo_, hi_) in lo_hi.items():
            out[f"split{key.capitalize()}{n.capitalize()}"] = \
                f"{m(lo_, 3)} to {m(hi_, 3)}" if key == "fweb" else f"{lo_:.3f}--{hi_:.3f}"
    out["splitProb"] = f"{2 / math.comb(len(SEEDS2), len(FRESH)):.1f}"
    rmean, rsd, rlo, rhi = interval(r2, T95)
    _, _, rlo90, rhi90 = interval(r2, T90)
    pmean, psd, plo, phi = interval(pooled, T95)
    _, _, plo90, phi90 = interval(pooled, T90)
    rmatched = [rows["python", s]["matched"][str(rep["r2"]["decision"]["common_target_update"])]["differential_forgetting"]
                for s in FRESH]
    out.update({"repDmean": m(rmean), "repDsd": m(rsd, sign=False), "repDlo": m(rlo), "repDhi": m(rhi),
                "repDloNinety": m(rlo90), "repDhiNinety": m(rhi90),
                "repDvalues": ", ".join(m(rows["python", s]["D"]) for s in FRESH),
                "repNpos": f"{sum(x > 0 for x in r2)}",
                "repDthree": m(statistics.mean(rows["python", s]["D3050"] for s in FRESH)),
                "repQmean": m(statistics.mean(rows["python", s]["Q"] for s in FRESH)),
                "repMatched": m(statistics.mean(rmatched)),
                "repGapMin": f"{min(rows['python', s]['gap'] for s in FRESH) * 100:.2f}",
                "repGapMax": f"{max(rows['python', s]['gap'] for s in FRESH) * 100:.2f}",
                "repAdaptMin": f"{min(v for s in FRESH for v in rows['python', s]['adapt'].values()):.2f}",
                "repAdaptMax": f"{max(v for s in FRESH for v in rows['python', s]['adapt'].values()):.2f}",
                "poolMean": m(pmean), "poolSd": m(psd, sign=False), "poolLo": m(plo), "poolHi": m(phi),
                "poolSdRatio": f"{psd / pilot_sd:.0f}", "poolLoNinety": m(plo90), "poolHiNinety": m(phi90)})
    R, T = ("x", "RMS"), ("x", "Taper-minus")
    out.update({"mcGapXR": f"{scale[R]['gap']:.3f}", "mcGapXT": f"{scale[T]['gap']:.3f}",
                "mcGapPyR": f"{scale['python', 'RMS']['gap']:.3f}", "mcGapPyT": f"{scale['python', 'Taper-minus']['gap']:.3f}",
                "mcSpecXR": f"{scale[R]['specific']:.3f}", "mcSpecXT": f"{scale[T]['specific']:.3f}",
                "mcSpecPyR": f"{scale['python', 'RMS']['specific']:.3f}",
                "mcSpecPyT": f"{scale['python', 'Taper-minus']['specific']:.3f}",
                "mcRatioLo": f"{min(scale[R]['specific'] / scale['python', 'RMS']['specific'], scale[T]['specific'] / scale['python', 'Taper-minus']['specific']):.1f}",
                "mcRatioHi": f"{max(scale[R]['specific'] / scale['python', 'RMS']['specific'], scale[T]['specific'] / scale['python', 'Taper-minus']['specific']):.1f}",
                "mcXfactorR": f"{math.exp(scale[R]['s_signed']):.2f}", "mcXfactorT": f"{math.exp(scale[T]['s_signed']):.2f}",
                "mcWebFactorR": f"{math.exp(scale[R]['w_signed']):.2f}", "mcWebFactorT": f"{math.exp(scale[T]['w_signed']):.2f}"})

    # ---- tables
    def table(path, lines):
        (OUT / path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    t = ["\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule",
         "Candidate & $S$ (both) & RMS & Taper-minus & First half & Second half \\\\", "\\midrule"]
    labels = {"mc4-zh": "Chinese web text (mC4 zh)", "mc4-ru": "Russian web text (mC4 ru)",
              "mc4-de": "German web text (mC4 de)", "openwebmath": "Mathematical web text (OpenWebMath)"}
    for cid in sel["ranking"]:
        c = cand[cid]
        t.append(f"{labels[cid]}{' (selected)' if cid == sel['selected'] else ''} & ${c['S']:.4f}$ & "
                 f"${c['per_condition']['RMS']:.4f}$ & ${c['per_condition']['Taper-minus']:.4f}$ & "
                 f"${c['half_sample_S'][0]:.4f}$ & ${c['half_sample_S'][1]:.4f}$ \\\\")
    py = sel["python_reference"]
    t += ["\\midrule", f"Python (pilot's code, reference) & ${py['S']:.4f}$ & ${py['per_condition']['RMS']:.4f}$ & "
          f"${py['per_condition']['Taper-minus']:.4f}$ & & \\\\", "\\bottomrule", "\\end{tabular}"]
    table("table_s2_selection.tex", t)

    t = ["\\begin{tabular}{@{}lrrrrrrrr@{}}", "\\toprule",
         "Seed & $D_X$ & $D_X(3050)$ & $D_X(1525)$ & $G_{\\text{web}}$ & $Q$ & Matched ($s{=}" + f"{target}" + "$) & "
         "Gap & $X$ gain (R / T) \\\\", "\\midrule"]
    for seed in SEEDS2:
        r = rows["x", seed]
        dagger = "$^\\dagger$" if seed in SEEDS else ""
        t.append(f"{seed}{dagger} & {fmt(r['D'])} & {fmt(r['D3050'])} & {fmt(r['D1525'])} & "
                 f"{fmt(r['G'])} & {fmt(r['Q'])} & {fmt(r['matched'][str(target)]['differential_forgetting'])} & "
                 f"${r['gap'] * 100:.2f}\\%$ & ${r['adapt']['RMS']:.2f}$ / ${r['adapt']['Taper-minus']:.2f}$ \\\\")
    means = {k: statistics.mean(rows["x", s][k] for s in SEEDS2) for k in ("D", "D3050", "D1525", "G", "Q")}
    sds = {k: statistics.stdev(rows["x", s][k] for s in SEEDS2) for k in ("D", "D3050", "D1525", "G", "Q")}
    t += ["\\midrule", "Mean & " + " & ".join(fmt(means[k]) for k in ("D", "D3050", "D1525", "G", "Q"))
          + f" & {fmt(statistics.mean(matched))} & & \\\\",
          "SD & " + " & ".join(f"${sds[k]:.4f}$" for k in ("D", "D3050", "D1525", "G", "Q"))
          + f" & ${statistics.stdev(matched):.4f}$ & & \\\\", "\\bottomrule", "\\end{tabular}"]
    table("table_s2_primary.tex", t)

    t = ["\\begin{tabular}{@{}lrrrrrrr@{}}", "\\toprule",
         "Seed & $D$ & $D(3050)$ & $G_{\\text{web}}$ & $Q$ & Matched ($s{=}6104$) & Gap & Code gain (R / T) \\\\",
         "\\midrule"]
    for seed in FRESH:
        r = rows["python", seed]
        t.append(f"{seed} & {fmt(r['D'])} & {fmt(r['D3050'])} & {fmt(r['G'])} & {fmt(r['Q'])} & "
                 f"{fmt(r['matched']['6104']['differential_forgetting'])} & ${r['gap'] * 100:.2f}\\%$ & "
                 f"${r['adapt']['RMS']:.2f}$ / ${r['adapt']['Taper-minus']:.2f}$ \\\\")
    rmeans = {k: statistics.mean(rows["python", s][k] for s in FRESH) for k in ("D", "D3050", "G", "Q")}
    rsds = {k: statistics.stdev(rows["python", s][k] for s in FRESH) for k in ("D", "D3050", "G", "Q")}
    t += ["\\midrule", "Mean & " + " & ".join(fmt(rmeans[k]) for k in ("D", "D3050", "G", "Q"))
          + f" & {fmt(statistics.mean(rmatched))} & & \\\\",
          "SD & " + " & ".join(f"${rsds[k]:.4f}$" for k in ("D", "D3050", "G", "Q"))
          + f" & ${statistics.stdev(rmatched):.4f}$ & & \\\\", "\\bottomrule", "\\end{tabular}"]
    table("table_s2_replication.tex", t)

    t = ["\\begin{tabular}{@{}llrrrrc@{}}", "\\toprule",
         "Contrast & Seeds & Mean & SD & 95\\% $t$-interval & 90\\% $t$-interval & Equivalent \\\\", "\\midrule"]
    blocks = [("Chinese, $D_X$ (H2)", "101--106", h2),
              ("\\quad fresh seeds only", "104--106", [rows["x", s]["D"] for s in FRESH]),
              ("Python, $D$ (R2)", "104--106", r2),
              ("\\quad pilot (Study 1)", "101--103", [rows["python", s]["D"] for s in SEEDS]),
              ("\\quad pooled (descriptive)", "101--106", pooled)]
    iv = lambda a, b: f"$[{a:+.4f}, {b:+.4f}]$".replace("+", "{+}")
    for name, seeds_txt, values in blocks:
        mean_, sd_, lo_, hi_ = interval(values, T95)
        _, _, lo9, hi9 = interval(values, T90)
        inside = "yes" if -0.015 < lo9 and hi9 < 0.015 else "no"
        t.append(f"{name} & {seeds_txt} & {fmt(mean_)} & ${sd_:.4f}$ & {iv(lo_, hi_)} & {iv(lo9, hi9)} & {inside} \\\\")
    t += ["\\bottomrule", "\\end{tabular}"]
    table("table_s2_intervals.tex", t)

    fs = lambda x: f"${x:+.3f}$ ($\\times{math.exp(x):.2f}$)".replace("+", "{+}")
    t = ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
         " & \\multicolumn{2}{c}{Chinese ($X$)} & \\multicolumn{2}{c}{Python} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
         "Log-ratio of scales & RMS & Taper-minus & RMS & Taper-minus \\\\",
         "\\midrule",
         "Switch gap, domain vs.\\ web, $|\\cdot|$ & " + " & ".join(f"{scale[k]['gap']:.3f}" for k in
                                                                     [R, T, ("python", "RMS"), ("python", "Taper-minus")]) + " \\\\",
         "Web branch: web-probe change, $|\\cdot|$ & " + " & ".join(f"{scale[k]['w_abs']:.3f}" for k in
                                                                      [R, T, ("python", "RMS"), ("python", "Taper-minus")]) + " \\\\",
         "Shifted branch: web-probe change, $|\\cdot|$ & " + " & ".join(f"{scale[k]['s_abs']:.3f}" for k in
                                                                          [R, T, ("python", "RMS"), ("python", "Taper-minus")]) + " \\\\",
         "\\quad signed (factor) & " + " & ".join(fs(scale[k]["s_signed"]) for k in
                                                 [R, T, ("python", "RMS"), ("python", "Taper-minus")]) + " \\\\",
         "Domain-specific change, $|\\cdot|$ & " + " & ".join(
             f"\\textbf{{{scale[k]['specific']:.3f}}}" for k in [R, T, ("python", "RMS"), ("python", "Taper-minus")]) + " \\\\",
         "\\bottomrule", "\\end{tabular}"]
    table("table_s2_scales.tex", t)

    t = ["\\begin{tabular}{@{}lrrrrrrrr@{}}", "\\toprule",
         " & \\multicolumn{4}{c}{Contribution to $D_X$ by token class} & \\multicolumn{3}{c}{Per-document $D_i$} & R \\\\",
         "\\cmidrule(lr){2-5}\\cmidrule(lr){6-8}",
         "Seed & W & A & P & C & Median & 1\\%-trimmed & Top-1\\% sum & contribution \\\\", "\\midrule"]
    for seed in SEEDS2:
        r = rows["x", seed]
        cls = {c["token_class"]: c for c in r["classes"]}
        t.append(f"{seed} & " + " & ".join(fmt(cls[c]["weighted_contribution"]) for c in "WAPX") + " & "
                 f"{fmt(r['tail']['unweighted_median'])} & {fmt(r['tail']['trimmed_token_weighted_mean'])} & "
                 f"{fmt(r['tail']['top_signed_sum'])} & {fmt(r['rare']['contribution_to_d'])} \\\\")
        assert abs(sum(c["weighted_contribution"] for c in r["classes"]) - r["D"]) < 1e-6
    t += ["\\bottomrule", "\\end{tabular}"]
    table("table_s2_tails.tex", t)

    phases = [("prefix_calibration", "Prefix, $u \\le 763$ (calibration)"),
              ("prefix_gate_decay", "Prefix, $764 \\le u \\le 6{,}103$"), ("prefix_gate_zero", "Prefix, $u \\ge 6{,}104$"),
              ("web", "Web branch"), ("python", "Python branch"), ("x", "Chinese branch")]
    t = ["\\begin{tabular}{@{}lrcc@{}}", "\\toprule", "Phase & Runs & RMS & Taper-minus \\\\", "\\midrule"]
    for key, name in phases:
        cells = [f"{clip2[c][key]['mean'] * 100:.1f}\\% ({clip2[c][key]['min'] * 100:.1f}--{clip2[c][key]['max'] * 100:.1f})"
                 for c in CONDS]
        t.append(f"{name} & {clip2['RMS'][key]['runs']} & " + " & ".join(cells) + " \\\\")
    t += ["\\bottomrule", "\\end{tabular}"]
    table("table_s2_clipping.tex", t)

    # ---- figure: Study 2 trajectories
    figure_style()
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

    ax = axes[0]
    for cond in CONDS:
        for branch, ls in (("x", "-"), ("web", (0, (3, 1.6)))):
            ys = [statistics.mean(ce2(s, cond, branch, u) for s in SEEDS2) for u in xs]
            ax.plot(xs, ys, color=COLOR[cond], linestyle=ls, linewidth=1.4, label=cond if branch == "x" else None)
    style(ax, "(a) Held-out web CE")
    ax.set_ylabel("Full-dev web CE (nats/token)")
    ax.legend(loc="center right", frameon=False, handlelength=1.8)
    ax.text(6104, 6.4, "Chinese branches (solid)", color=INK2, fontsize=6.2, ha="right", va="top")
    ax.text(6104, 4.66, "web branches (dashed)", color=INK2, fontsize=6.2, ha="right", va="bottom")
    ax = axes[1]
    for y in (0.03, 0.015, -0.03):
        ax.axhline(y, color=MUTED, linewidth=0.6)
    ax.axhline(0, color=INK2, linewidth=0.7)
    for seed in SEEDS2:
        ax.plot(xs, [rows["x", seed]["full_D"][u] for u in xs], color=MUTED, linewidth=0.7,
                linestyle="-" if seed in FRESH else (0, (2, 1.2)))
    ax.plot([], [], color=MUTED, linewidth=0.7, linestyle=(0, (2, 1.2)), label="101–103")
    ax.plot([], [], color=MUTED, linewidth=0.7, label="104–106")
    ax.plot(xs, [statistics.mean(rows["x", s]["full_D"][u] for s in SEEDS2) for u in xs], color=INK, linewidth=1.6,
            label="mean")
    style(ax, "(b) Contrast $D_X(s)$, full dev")
    ax.grid(False, axis="y")
    lim = max(abs(rows["x", s]["full_D"][u]) for s in SEEDS2 for u in xs)
    assert lim < 0.235  # one seed-point excursion (seed 104, s = 3,355) reaches about -0.22
    ax.set_ylim(-0.235, 0.09)
    ax.set_yticks([-0.2, -0.15, -0.1, -0.05, 0, 0.05])
    ax.set_yticklabels(["−0.20", "−0.15", "−0.10", "−0.05", "0", "+0.05"])
    ax.set_ylabel("$D_X(s)$ (nats/token)")
    ax.legend(loc="lower center", frameon=False, handlelength=1.3, fontsize=5.6, ncol=3, columnspacing=0.7,
              title="seeds", title_fontsize=5.6, borderaxespad=0.2)
    ax = axes[2]
    for cond in CONDS:
        ys = [statistics.mean(ce2(s, cond, "x", u, dom="x", metric="non_w_ce") for s in SEEDS2) for u in xs]
        ax.plot(xs, ys, color=COLOR[cond], linewidth=1.4, label=cond)
    style(ax, "(c) Chinese adaptation")
    ax.set_ylabel("Full-dev non-W Chinese CE (nats/token)")
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(FIG / "study2_trajectories.pdf", metadata={"CreationDate": None})
    fig.savefig(FIG / "study2_trajectories.png", dpi=220)
    plt.close(fig)
    return out, {"h2": (mean, lo, hi), "h2_hi90": hi90, "r2": rmean, "ratio": (out["mcRatioLo"], out["mcRatioHi"]),
                 "ctrl_share": float(out["twoCtrlShare"])}


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

    def fmt(x, d=4):
        return f"${x:+.{d}f}$".replace("+", "{+}")

    two, s2 = study2_assets(ev, m, fmt)
    macros.update(two)

    # ---- abstract: one ASCII text for both the PDF and the arXiv metadata field
    (h2_mean, h2_lo, h2_hi), r2_mean, h2_hi90, ctrl = s2["h2"], s2["r2"], s2["h2_hi90"], s2["ctrl_share"]
    ratio = round(float(s2["ratio"][0]))
    assert round(float(s2["ratio"][1])) == ratio
    abstract = (
        "TaperNorm replaces a pre-norm Transformer's internal normalization with a gated map that acts like "
        "RMSNorm early in training and then becomes a fixed linear scaling. Without per-token normalization, a "
        "model might forget more after a shift in the training data. Two small studies tested this, with designs, "
        "endpoints and decision rules fixed before training. Paired 17.7M-parameter models with "
        "internal RMSNorm or TaperNorm were trained on 150M web tokens, then continued for 100M tokens on web "
        "text or a new domain. The endpoint D is a difference-in-differences in held-out web cross-entropy; "
        "positive D means extra forgetting under TaperNorm. In a pilot with Python code (three seeds), mean D "
        f"was {mean_d:.3f} nats/token, below the pre-specified +0.015 bound, so the decision was to stop. A second "
        "study, whose frozen protocol and code were hashed and submitted for timestamping before it ran, chose a "
        "new domain by a fixed rule (Chinese web text) and added three fresh seeds. On Chinese, mean D over six "
        f"seeds was {h2_mean:.3f} (95% interval {h2_lo:.3f} to {h2_hi:+.3f}), just past the -0.03 threshold for "
        f"the opposite direction; a post hoc one-sided 95% upper bound of {h2_hi90:.3f} rules out an excess of +0.015. "
        f"Under GPT-2's tokenizer, however, {ctrl:.0f}% of Chinese labels are byte fragments, so much of that shift "
        "concerns which output tokens must be predicted, which the internal normalizers do not touch. "
        "On the fresh seeds "
        f"the Python decision replicated (mean D {r2_mean:+.3f}), but D was positive in all three, where it had "
        "been negative in all three pilot seeds: three seeds understated seed variation. A pre-specified "
        "in-training measure of the scale shift behind the hypothesis was about "
        f"{['zero', 'one', 'two', 'three', 'four', 'five', 'six'][ratio]} times larger for Chinese than for "
        "Python, yet TaperNorm did not forget more. Neither study showed equivalence within +/-0.015. "
        "We find no evidence that TaperNorm increases forgetting under these shifts. Code and all records are "
        "released.")
    assert h2_hi90 < -0.0 and r2_mean > 0, "abstract's one-sided-bound and sign statements"
    assert macros["repNpos"] == "3" and all(rows[s]["D"] < 0 for s in SEEDS), "abstract's sign statement"
    assert macros["repInPilot"] == "0", "main text: no fresh value lies inside Study 1's interval"
    assert abstract.isascii(), "abstract must be plain ASCII"
    assert len(abstract) <= 1920, f"arXiv abstract limit: {len(abstract)} characters"
    (ROOT / "paper" / "arxiv-abstract.txt").write_text(abstract + "\n", encoding="ascii")
    tex_abstract = re.sub(r"(?<=[\s(])-(?=\d)", "$-$", abstract.replace("%", "\\%").replace("+/-", "$\\pm$"))
    (OUT / "abstract.tex").write_text(tex_abstract + "\n", encoding="utf-8")
    print(f"abstract: {len(abstract)} characters, ASCII")
    lines = ["% Generated by paper/make_assets.py from reports/h1-kaggle and reports/s2-kaggle -- do not edit by hand."]
    # Thousands separators as {,} so numbers typeset correctly in both text and math mode.
    lines += [f"\\newcommand{{\\{k}}}{{{v.replace(',', '{,}')}}}" for k, v in macros.items()]
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

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
         "Seed & W & A & P & C & Median & 1\\%-trimmed & Top-1\\% sum \\\\", "\\midrule"]
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
    figure_style()
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
    print("independent recomputation (Studies 1 and 2) agrees with the frozen reports; wrote", len(macros), "macros, 11 tables, 2 figures")
    for k in ("Dmean", "Dsd", "Dlo", "Dhi", "DthreeMean", "Qmean", "Gmean", "matchedMean", "FcodeMean", "DrelPct",
              "earlyMax", "earlyMaxAt", "clipRMSlo", "clipRMShi", "clipTaperlo", "clipTaperhi", "paramsRMS",
              "embedShare", "nDocsWeb", "slopeRMS", "slopeTaper", "sessionStart", "sessionEnd"):
        print(f"  {k} = {macros[k]}")


if __name__ == "__main__":
    main()
