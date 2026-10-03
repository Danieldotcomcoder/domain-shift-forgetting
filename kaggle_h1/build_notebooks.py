"""Build the Kaggle notebooks (+ kernel-metadata.json) from h1_run.py.

  python kaggle_h1/build_notebooks.py run --phase bootstrap   # first push only (CPU, seconds)
  python kaggle_h1/build_notebooks.py run --phase gpu         # every later push (T4 x2, resumes itself)
  python kaggle_h1/build_notebooks.py smoke                   # separate short GPU smoke test
Then:  kaggle kernels push -p kaggle_h1/kernel-run   (or kernel-smoke)
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
USER = "danny00"
DATASET = f"{USER}/stage1-online-inputes"
RUN_SLUG = "h1-single-notebook-run"
SMOKE_SLUG = "h1-single-notebook-smoke"


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(keepends=True)}


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def notebook(cells: list) -> dict:
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index}"
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                        "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


RUNNER = (HERE / "h1_run.py").read_text(encoding="utf-8")
WRITE_RUNNER = "%%writefile /kaggle/working/h1_run.py\n" + RUNNER

INTRO = """# H1 — does TaperNorm increase persistent web forgetting after a web → Python switch?

**How to use:** *Save Version → Save & Run All (Commit)* with accelerator **GPU T4 x2**. Nothing to fill in.

* Each run trains for up to ~11¼ hours, saves everything into this notebook's own **Output** (`h1state/`), and stops cleanly.
* The next run receives that Output as **Input** automatically and resumes exactly where it stopped
  (RMS on GPU 0, Taper‑minus on GPU 1; seeds 101 → 102 → 103; prefix → web branch → Python branch).
* When all six runs are complete it prints the pre‑registered decision and the **H1 answer**
  (also in `h1state/decision-report.txt` / `.json`). Re‑running a finished experiment does nothing.

Protocol v3 (H1 only, primary allocation) with the recorded T4/FP16 amendment. The frozen configuration,
data hashes and amendments are in `h1state/config.json`.
"""

LAUNCH = """import os, subprocess, sys, time
env = dict(os.environ, H1_T0=str(T0), H1_SESSION_HOURS=str(SESSION_HOURS), PYTHONUNBUFFERED="1")
proc = subprocess.Popen([sys.executable, "-u", "/kaggle/working/h1_run.py", "main"], env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    print(line, end="", flush=True)
print("runner exit code:", proc.wait())
"""

SHOW = """from pathlib import Path
report = Path("/kaggle/working/h1state/decision-report.txt")
print(report.read_text() if report.exists() else "No report yet (bootstrap run or setup error above).")
"""


def build_run(phase: str) -> Path:
    folder = HERE / "kernel-run"
    folder.mkdir(exist_ok=True)
    cells = [markdown(INTRO),
             code("import time\nT0 = time.time()          # session clock starts here\nSESSION_HOURS = 11.25  # Kaggle limit is 12 h; keep the margin\n"),
             code(WRITE_RUNNER), code(LAUNCH), code(SHOW)]
    (folder / f"{RUN_SLUG}.ipynb").write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
    gpu = phase == "gpu"
    meta = {"id": f"{USER}/{RUN_SLUG}", "title": RUN_SLUG, "code_file": f"{RUN_SLUG}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": gpu, "enable_tpu": False, "enable_internet": False,
            "dataset_sources": [DATASET], "competition_sources": [], "model_sources": [],
            "kernel_sources": [f"{USER}/{RUN_SLUG}"] if gpu else []}
    if gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return folder


SMOKE = r'''import json, os, shutil, subprocess, sys, time
from pathlib import Path
online = [p.parent for p in Path("/kaggle/input").rglob("manifest.json")
          if p.parent.name == "online" and json.loads(p.read_text()).get("schema") == "pilot-arrays-v1"][0]
orders = online.parent / "orders"
RUNNER = "/kaggle/working/h1_run.py"
results = {}

def run(work, inputs, hours, extra_env=None, extra_args=()):
    Path(inputs).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, H1_T0=str(time.time()), H1_SESSION_HOURS=str(hours), PYTHONUNBUFFERED="1",
               **(extra_env or {}))
    command = [sys.executable, "-u", RUNNER, "main", "--inputs", str(inputs), "--work", str(work),
               "--online", str(online), "--orders", str(orders), *extra_args]
    proc = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    for line in proc.stdout:
        lines.append(line)
        print(line, end="", flush=True)
    return proc.wait(), "".join(lines)

W = Path("/tmp/smoke"); shutil.rmtree(W, ignore_errors=True); W.mkdir(parents=True)

# A) Mini schedule end-to-end on both GPUs, interrupted twice, resumed from previous output each time.
print("\n########## A) MINI PROTOCOL END-TO-END (plumbing only) ##########", flush=True)
code1, out1 = run(W / "m1", W / "empty", 0.03, {"H1_ALLOW_FRESH_START": "1"}, ["--mini"])
code2, out2 = run(W / "m2", W / "m1", 0.03, None, ["--mini"])
code3, out3 = run(W / "m3", W / "m2", 0.5, None, ["--mini"])
code4, out4 = run(W / "m4", W / "m3", 0.5, None, ["--mini"])
rep = json.loads((W / "m3/h1state/decision-report.json").read_text())
results["mini"] = {"exit_codes": [code1, code2, code3, code4], "complete": rep["complete"],
                   "category": rep["decision"]["category"], "seed_groups": len(rep["seeds"]),
                   "failures": rep["failures"], "rerun_noop": "Nothing to train" in out4,
                   "sessions": len((W / "m3/h1state/sessions.jsonl").read_text().splitlines())}
print("MINI RESULT:", json.dumps(results["mini"]), flush=True)

# B) Real protocol: a fresh 15-minute session, then a 6-minute resumed session (throughput + real resume).
print("\n########## B) REAL PROTOCOL THROUGHPUT + RESUME ##########", flush=True)
codeb1, _ = run(W / "r1", W / "empty", 0.25, {"H1_ALLOW_FRESH_START": "1"})
codeb2, _ = run(W / "r2", W / "r1", 0.20)
summary = {"exit_codes": [codeb1, codeb2]}
for condition in ("RMS", "Taper-minus"):
    run_dir = W / "r2/h1state/runs" / f"S101-{condition}"
    rows = [json.loads(l) for l in (run_dir / "train.jsonl").read_text().splitlines() if l.strip()]
    events = [json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines() if l.strip()]
    def mean_sec(lo, hi):
        values = [r["sec"] for r in rows if lo <= r["u"] <= hi]
        return (sum(values) / len(values), len(values)) if values else (None, 0)
    summary[condition] = {
        "last_update": max(r["u"] for r in rows),
        "sec_per_update_calibration": mean_sec(20, 763), "sec_per_update_after": mean_sec(764, 99999),
        "max_overflow_retries": max(r["retry"] for r in rows), "final_scale": rows[-1]["scale"],
        "eval_seconds": {f"{e['role']}-{e['domain']}": round(e["seconds"], 1) for e in events},
        "full_dev_ce": {f"{e['step']}-{e['domain']}": e.get("ce") for e in events if e["role"] == "full"}}
summary["resume_lines"] = sum(p.read_text().count("resumed at") for p in (W / "r2/h1state/logs").glob("*.log"))
results["real"] = summary
print("REAL RESULT:", json.dumps(summary, indent=1), flush=True)
Path("/kaggle/working/smoke-results.json").write_text(json.dumps(results, indent=1))
# keep small logs only
for d in ("m3", "r2"):
    shutil.copytree(W / d / "h1state", Path("/kaggle/working") / f"{d}-h1state",
                    ignore=shutil.ignore_patterns("*.pt", "*.pt.tmp"))
'''


def build_smoke() -> Path:
    folder = HERE / "kernel-smoke"
    folder.mkdir(exist_ok=True)
    cells = [markdown("# H1 runner smoke test (separate notebook; not the experiment)\n"),
             code(WRITE_RUNNER), code(SMOKE)]
    (folder / f"{SMOKE_SLUG}.ipynb").write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
    meta = {"id": f"{USER}/{SMOKE_SLUG}", "title": SMOKE_SLUG, "code_file": f"{SMOKE_SLUG}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_tpu": False, "enable_internet": False, "dataset_sources": [DATASET],
            "competition_sources": [], "model_sources": [], "kernel_sources": [],
            "machine_shape": "NvidiaTeslaT4"}
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("which", choices=["run", "smoke"])
    parser.add_argument("--phase", choices=["bootstrap", "gpu"], default="gpu")
    args = parser.parse_args()
    print(build_run(args.phase) if args.which == "run" else build_smoke())
