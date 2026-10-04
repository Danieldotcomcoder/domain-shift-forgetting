"""Generate every number, table and figure in the paper from the run's own records.

Reads reports/h1-kaggle/ (decision report, config, session record, per-run evaluation logs,
completion receipts, and corpus/{audit,manifest}.json). Recomputes the primary
contrasts independently from the raw evaluation records and asserts agreement with the frozen
analysis output before writing anything. Nothing in the manuscript is typed by hand.

  python paper/make_assets.py
"""
import json
import math
import statistics
import datetime as dt
from pathlib import Path

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
        "sessionStart": start.strftime("%Y-%m-%d %H:%M"), "sessionEnd": end.strftime("%Y-%m-%d %H:%M"),
        "torchVersion": session["torch"].split("+")[0], "cudaVersion": session["cuda"],
        "paramsRMS": f"{params_rms:,}", "paramsTaper": f"{params_taper:,}",
        "embedShare": f"{emb / params_rms * 100:.0f}",
        "nDocsWeb": f"{n_docs_web:,}",
        "medianDocMin": m(min(rows[s]["tail"]["unweighted_median"] for s in SEEDS), 3),
        "medianDocMax": m(max(rows[s]["tail"]["unweighted_median"] for s in SEEDS), 3),
        "configSha": config["config_sha256"][:16],
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
    fig.savefig(FIG / "trajectories.pdf")
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
