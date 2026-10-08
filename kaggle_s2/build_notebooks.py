"""Build the Study 2 Kaggle notebooks (+ kernel-metadata.json), the private-dataset folder and the registration
manifest. Building pushes nothing; every Kaggle command is printed for the user to run (see kaggle_s2/README.md).

  python kaggle_s2/build_notebooks.py pin-image             # once: record the pilot's Docker image (Kaggle API read)
  python kaggle_s2/build_notebooks.py freeze                # code/protocol hashes for the public registration
  python kaggle_s2/build_notebooks.py probe                 # 1. domain-selection probe (GPU, internet on)
  python kaggle_s2/build_notebooks.py prepare               # 2. selected-domain arrays + orders (CPU, internet on)
  python kaggle_s2/build_notebooks.py dataset --prepared DIR    # 3. private dataset folder (no reserved test)
  python kaggle_s2/build_notebooks.py smoke                 # 4. smoke test (separate notebook; no outcome data)
  python kaggle_s2/build_notebooks.py run --phase bootstrap # 5. first push only (CPU, seconds)
  python kaggle_s2/build_notebooks.py run --phase gpu       #    every later push (T4 x2, resumes itself)
Then:  kaggle kernels push -p kaggle_s2/kernel-<name>
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
USER = "danny00"
PILOT_DATASET = f"{USER}/stage1-online-inputes"
S2_DATASET = f"{USER}/study2-online-inputs"
PILOT_RUN = f"{USER}/h1-single-notebook-run"
PROBE_SLUG = "s2-domain-probe"
PREPARE_SLUG = "s2-prepare-domain"
SMOKE_SLUG = "s2-single-notebook-smoke"
RUN_SLUG = "s2-single-notebook-run"
ENVIRONMENT = HERE / "environment.json"
KERNELS_DIR = HERE  # notebooks are written to KERNELS_DIR / "kernel-<name>" (tests redirect this)
REGISTRATION = ROOT / "study2" / "registration-manifest.json"
CODE_FILES = ("s2_run.py", "prepare_domain.py", "probe_domains.py", "build_notebooks.py", "s2_status.py",
              "test_s2_run.py")


def source(path: Path) -> str:
    """File text with LF line endings: exactly the bytes %%writefile recreates on Kaggle."""
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(source(path).encode("utf-8")).hexdigest()


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def writefile(name: str, path: Path) -> dict:
    return code(f"%%writefile /kaggle/working/{name}\n" + source(path))


def notebook(cells: list) -> dict:
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index}"
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                        "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def registration_cells() -> list:
    """Every notebook embeds the frozen registration manifest, so each Kaggle version (timestamped server-side) carries
    the protocol and code hashes; the scripts record its SHA-256 in their outputs."""
    if not REGISTRATION.exists():
        raise SystemExit("Run `build_notebooks.py freeze` (and commit) first: every notebook embeds the registration "
                         "manifest")
    text = source(REGISTRATION)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return [markdown(f"**Registration manifest** — SHA-256 `{digest}`: the frozen protocol and code hashes, "
                     "timestamped before any Study 2 run (study2/PROTOCOL.md, Amendment 1).\n"),
            code("%%writefile /kaggle/working/registration-manifest.json\n" + text)]


def docker_image() -> str | None:
    if ENVIRONMENT.exists():
        return json.loads(ENVIRONMENT.read_text(encoding="utf-8")).get("docker_image")
    return None


def write_kernel(folder_name: str, slug: str, cells: list, *, gpu: bool, internet: bool, datasets: list,
                 kernels: list, pin: bool) -> Path:
    folder = KERNELS_DIR / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{slug}.ipynb").write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
    meta = {"id": f"{USER}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
            "enable_internet": internet, "dataset_sources": datasets, "competition_sources": [],
            "model_sources": [], "kernel_sources": kernels}
    if gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    image = docker_image() if pin else None
    if image:
        meta["docker_image"] = image
    elif pin:
        print("NOTE: no kaggle_s2/environment.json; this GPU notebook will use Kaggle's current image "
              "(run `pin-image` first to reuse the pilot's).", file=sys.stderr)
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return folder


NO_GPU_CHECK = """import subprocess
probe = subprocess.run(['sh', '-c', 'nvidia-smi -L 2>/dev/null'], capture_output=True, text=True)
assert 'GPU 0:' not in probe.stdout, 'Set Accelerator = None: preparation is CPU-only.'
"""

PIP = "import subprocess, sys\nsubprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'tiktoken', " \
      "'datasketch'], check=True)\n"


def build_probe() -> Path:
    intro = """# Study 2 · step 1 — domain-selection probe (forward passes only)

Pre-registered rule (study2/PROTOCOL.md, Sec. 4): on the pilot's six switch states, measure the scale of the
activations entering the 12 internal normalizers on each candidate's probe sample and select the candidate whose
mean absolute log scale ratio to web text is largest. No training happens here.

**Settings:** Accelerator **GPU T4**, Internet **on**. Inputs: the pilot dataset and the pilot run notebook.
Output: `s2probe/selection.json` (+ `selection.txt`, probe samples).
"""
    run = """import subprocess, sys
proc = subprocess.run([sys.executable, '-u', '/kaggle/working/probe_domains.py', '--inputs', '/kaggle/input',
                       '--output', '/kaggle/working/s2probe', '--cache', '/tmp/s2probe-cache'])
print('probe exit code:', proc.returncode)
"""
    show = "print(open('/kaggle/working/s2probe/selection.txt').read())\n"
    cells = [markdown(intro), *registration_cells(), code(PIP), writefile("s2_run.py", HERE / "s2_run.py"),
             writefile("prepare_domain.py", HERE / "prepare_domain.py"),
             writefile("probe_domains.py", HERE / "probe_domains.py"), code(run), code(show)]
    return write_kernel("kernel-probe", PROBE_SLUG, cells, gpu=True, internet=True, datasets=[PILOT_DATASET],
                        kernels=[PILOT_RUN], pin=True)


def build_prepare(rank: int = 0) -> Path:
    """rank 0 = the probe's selection; rank 1 = the protocol's pre-registered fallback (Sec. 5.9), never anything else."""
    if rank not in (0, 1):
        raise ValueError("Only the selection (0) or the pre-registered fallback (1) may be prepared")
    intro = """# Study 2 · step 2 — prepare the selected domain (CPU only)

Streams the domain chosen by the probe from Hugging Face at its pinned revision, GPT-2-tokenizes it, removes exact
and near duplicates (the pilot's frozen corpus always wins), splits and packs it exactly like the pilot, generates
the training orders and runs the acceptance checks. Raw text is never stored.

**Settings:** Accelerator **None**, Internet **on**. Inputs: the pilot dataset and the `s2-domain-probe` notebook.
Afterwards download the output and create the private dataset (kaggle_s2/README.md, step 3). The reserved test
(`reserved-test/`) must never be attached to a training notebook.
"""
    run = f"""import subprocess, sys
proc = subprocess.run([sys.executable, '-u', '/kaggle/working/prepare_domain.py', 'prepare', '--inputs',
                       '/kaggle/input', '--output', '/kaggle/working/s2prep', '--cache', '/tmp/s2cache',
                       '--rank', '{rank}'])
print('preparation exit code:', proc.returncode)
"""
    show = "print(open('/kaggle/working/s2prep/acceptance.json').read())\n"
    cells = [markdown(intro), *registration_cells(), code(NO_GPU_CHECK), code(PIP),
             writefile("prepare_domain.py", HERE / "prepare_domain.py"), code(run), code(show)]
    return write_kernel("kernel-prepare", PREPARE_SLUG, cells, gpu=False, internet=True, datasets=[PILOT_DATASET],
                        kernels=[f"{USER}/{PROBE_SLUG}"], pin=False)


RUN_INTRO = """# Study 2 — does TaperNorm forget more after a switch to the selected domain? (+ web→Python replication)

**How to use:** *Save Version → Save & Run All (Commit)* with accelerator **GPU T4 x2**. Nothing to fill in.

* Each run trains for up to ~11¼ hours, saves everything into this notebook's own **Output** (`s2state/`), and stops.
* The next run receives that Output as **Input** automatically and resumes exactly where it stopped (RMS on GPU 0,
  Taper‑minus on GPU 1; seeds 101–103: X branch from the pilot's switch states; seeds 104–106: prefix → web →
  Python → X).
* When all twelve runs are complete it prints the pre‑registered decisions (**H2** and **R2**), also in
  `s2state/decision-report.txt` / `.json`. Re‑running a finished experiment does nothing.

Pre-registered protocol: study2/PROTOCOL.md. The frozen configuration and data hashes are in `s2state/config.json`.
"""

LAUNCH = """import os, subprocess, sys, time
env = dict(os.environ, S2_T0=str(T0), S2_SESSION_HOURS=str(SESSION_HOURS), PYTHONUNBUFFERED="1")
proc = subprocess.Popen([sys.executable, "-u", "/kaggle/working/s2_run.py", "main"], env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    print(line, end="", flush=True)
print("runner exit code:", proc.wait())
"""

SHOW = """from pathlib import Path
report = Path("/kaggle/working/s2state/decision-report.txt")
print(report.read_text() if report.exists() else "No report yet (bootstrap run or setup error above).")
"""


def build_run(phase: str, session_hours: float = 11.25) -> Path:
    """``session_hours`` < 11.25 only to fit the remaining weekly GPU quota (the runner then stops and saves early)."""
    if not 0.5 <= session_hours <= 11.25:
        raise ValueError("A session must be between 0.5 and 11.25 hours (Kaggle's limit is 12 h)")
    gpu = phase == "gpu"
    cells = [markdown(RUN_INTRO), *registration_cells(),
             code("import time\nT0 = time.time()          # session clock starts here\n"
                  f"SESSION_HOURS = {session_hours}  # Kaggle limit is 12 h; keep the margin\n"),
             writefile("s2_run.py", HERE / "s2_run.py"), code(LAUNCH), code(SHOW)]
    return write_kernel("kernel-run", RUN_SLUG, cells, gpu=gpu, internet=False,
                        datasets=[PILOT_DATASET, S2_DATASET],
                        kernels=[PILOT_RUN, f"{USER}/{RUN_SLUG}"] if gpu else [], pin=gpu)


SMOKE = r'''import json, os, shutil, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, "/kaggle/working")
import s2_run as S2
paths = S2.locate_inputs(Path("/kaggle/input"))
H1, RUNNER = "/kaggle/working/h1_run.py", "/kaggle/working/s2_run.py"
DATA = ["--pilot-online", str(paths["pilot_online"]), "--pilot-orders", str(paths["pilot_orders"]),
        "--s2-online", str(paths["s2_online"]), "--s2-orders", str(paths["s2_orders"])]
results = {}

def run(command, extra_env):
    env = dict(os.environ, PYTHONUNBUFFERED="1", **extra_env)
    proc = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    for line in proc.stdout:
        lines.append(line)
        print(line, end="", flush=True)
    return proc.wait(), "".join(lines)

W = Path("/tmp/s2smoke"); shutil.rmtree(W, ignore_errors=True); (W / "empty").mkdir(parents=True)

# A) Mini protocol (plumbing only): a mini "pilot" made by the pilot's own runner, then Study 2 on top of it,
#    interrupted twice and resumed from the previous output each time.
print("\n########## A) MINI PROTOCOL END-TO-END (plumbing only) ##########", flush=True)
code0, _ = run([sys.executable, "-u", H1, "main", "--mini", "--inputs", str(W / "empty"), "--work", str(W / "pilot"),
                "--online", str(paths["pilot_online"]), "--orders", str(paths["pilot_orders"])],
               {"H1_T0": str(time.time()), "H1_SESSION_HOURS": "0.5", "H1_ALLOW_FRESH_START": "1"})
MINI_PILOT = ["--pilot-state", str(W / "pilot" / "h1state")]

def mini(work, inputs, hours, extra=None):
    return run([sys.executable, "-u", RUNNER, "main", "--mini", "--inputs", str(inputs), "--work", str(work),
                *DATA, *MINI_PILOT], {"S2_T0": str(time.time()), "S2_SESSION_HOURS": str(hours), **(extra or {})})

code1, out1 = mini(W / "m1", W / "empty", 0.03, {"S2_ALLOW_FRESH_START": "1"})
code2, out2 = mini(W / "m2", W / "m1", 0.03)
code3, out3 = mini(W / "m3", W / "m2", 0.5)
code4, out4 = mini(W / "m4", W / "m3", 0.5)
rep = json.loads((W / "m3/s2state/decision-report.json").read_text())
results["mini"] = {"pilot_exit": code0, "exit_codes": [code1, code2, code3, code4], "complete": rep["complete"],
                   "h2": rep["h2"]["decision"]["category"], "r2": rep["r2"]["decision"]["category"],
                   "h2_seed_groups": len(rep["h2"]["seeds"]), "failures": rep["failures"], "problems": rep["problems"],
                   "lineage": rep["lineage"], "rerun_noop": "Nothing to train" in out4,
                   "sessions": len((W / "m3/s2state/sessions.jsonl").read_text().splitlines())}
print("MINI RESULT:", json.dumps(results["mini"], indent=1), flush=True)

# B) Lineage: re-evaluate the six REAL pilot switch states (forward passes only) against the pilot's records.
print("\n########## B) LINEAGE OF THE REAL PILOT SWITCH STATES ##########", flush=True)
codeL, _ = run([sys.executable, "-u", RUNNER, "lineage", "--inputs", "/kaggle/input", "--output",
                str(W / "lineage.json")], {})
results["lineage"] = json.loads((W / "lineage.json").read_text()) | {"exit": codeL}

# C) Real protocol, fresh seed 104's web PREFIX only (no branch, so no outcome data): throughput + resume.
print("\n########## C) REAL PROTOCOL THROUGHPUT + RESUME (seed 104 prefix only) ##########", flush=True)
REAL = [*DATA, "--pilot-state", str(paths["pilot_state"])]
codeC1, _ = run([sys.executable, "-u", RUNNER, "main", "--inputs", str(W / "empty"), "--work", str(W / "r1"), *REAL],
                {"S2_T0": str(time.time()), "S2_SESSION_HOURS": "0.25", "S2_ALLOW_FRESH_START": "1",
                 "S2_ONLY_SEEDS": "104"})
codeC2, _ = run([sys.executable, "-u", RUNNER, "main", "--inputs", str(W / "r1"), "--work", str(W / "r2"), *REAL],
                {"S2_T0": str(time.time()), "S2_SESSION_HOURS": "0.20", "S2_ONLY_SEEDS": "104"})
summary = {"exit_codes": [codeC1, codeC2]}
for condition in ("RMS", "Taper-minus"):
    run_dir = W / "r2/s2state/runs" / f"S104-{condition}"
    rows = [json.loads(l) for l in (run_dir / "train.jsonl").read_text().splitlines() if l.strip()]
    events = [json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines() if l.strip()]
    def mean_sec(lo, hi):
        values = [r["sec"] for r in rows if lo <= r["u"] <= hi]
        return (sum(values) / len(values), len(values)) if values else (None, 0)
    summary[condition] = {"last_update": max(r["u"] for r in rows),
                          "sec_per_update_calibration": mean_sec(20, 763), "sec_per_update_after": mean_sec(764, 99999),
                          "max_overflow_retries": max(r["retry"] for r in rows), "final_scale": rows[-1]["scale"],
                          "eval_seconds": {f"{e['role']}-{e['domain']}": round(e["seconds"], 1) for e in events
                                           if "seconds" in e}}
summary["resume_lines"] = sum(p.read_text().count("resumed at") for p in (W / "r2/s2state/logs").glob("*.log"))
results["real"] = summary
print("REAL RESULT:", json.dumps(summary, indent=1), flush=True)
Path("/kaggle/working/smoke-results.json").write_text(json.dumps(results, indent=1))
for name in ("m3", "r2"):  # keep small logs only
    shutil.copytree(W / name / "s2state", Path("/kaggle/working") / f"{name}-s2state",
                    ignore=shutil.ignore_patterns("*.pt", "*.pt.tmp"))
ok = (results["mini"]["complete"] and not results["mini"]["failures"] and not results["mini"]["problems"]
      and results["mini"]["rerun_noop"] and results["lineage"]["passed"] and summary["resume_lines"] >= 2)
print("SMOKE", "PASSED" if ok else "NEEDS ATTENTION", flush=True)
'''


def build_smoke() -> Path:
    intro = """# Study 2 runner smoke test (separate notebook; not the experiment; produces no outcome data)

A) mini protocol end-to-end on a mini "pilot" made by the pilot's own runner, with interruptions and resumes;
B) lineage of the six real pilot switch states (forward passes only); C) real-protocol throughput and resume on
seed 104's web prefix only. **Settings:** GPU T4 x2, Internet off.
"""
    cells = [markdown(intro), *registration_cells(), writefile("h1_run.py", ROOT / "kaggle_h1" / "h1_run.py"),
             writefile("s2_run.py", HERE / "s2_run.py"), code(SMOKE)]
    return write_kernel("kernel-smoke", SMOKE_SLUG, cells, gpu=True, internet=False,
                        datasets=[PILOT_DATASET, S2_DATASET], kernels=[PILOT_RUN], pin=True)


def kaggle(*args: str) -> subprocess.CompletedProcess:
    import os
    env = dict(os.environ, PYTHONUTF8="1")
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.split("=", 1)
                env.setdefault(key.strip(), value.strip())
    return subprocess.run([sys.executable, "-X", "utf8", "-m", "kaggle.cli", *args], capture_output=True, text=True,
                          encoding="utf-8", cwd=ROOT, env=env)


def pin_image() -> Path:
    """Read the pilot notebook's Docker image from Kaggle (metadata pull; nothing is pushed or run)."""
    with tempfile.TemporaryDirectory() as tmp:
        result = kaggle("kernels", "pull", PILOT_RUN, "-p", tmp, "-m")
        meta_path = Path(tmp) / "kernel-metadata.json"
        if not meta_path.exists():
            raise SystemExit(f"Could not pull the pilot's metadata:\n{result.stdout}\n{result.stderr}")
        image = json.loads(meta_path.read_text(encoding="utf-8")).get("docker_image")
    if not image:
        raise SystemExit("The pilot notebook's metadata has no docker_image; GPU notebooks will use the current image.")
    ENVIRONMENT.write_bytes((json.dumps({"docker_image": image, "source_kernel": PILOT_RUN,
                                         "retrieved_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                                         "pilot_stack": {"torch": "2.11.0+cu128", "cuda": "12.8",
                                                         "driver": "580.178.04", "python": "3.13.15"}},
                                        indent=2) + "\n").encode("utf-8"))
    return ENVIRONMENT


def freeze() -> Path:
    """SHA-256 (LF-normalized, i.e. as run on Kaggle) of the protocol and every Study 2 file, for the registration."""
    sys.path.insert(0, str(HERE))
    import prepare_domain as PD
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "kaggle_s2", "study2"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    files = {f"kaggle_s2/{name}": sha256_lf(HERE / name) for name in CODE_FILES}
    files["study2/PROTOCOL.md"] = sha256_lf(ROOT / "study2" / "PROTOCOL.md")
    files["kaggle_h1/h1_run.py (pilot runner, unchanged; used by the smoke test)"] = sha256_lf(
        ROOT / "kaggle_h1" / "h1_run.py")
    if ENVIRONMENT.exists():
        files["kaggle_s2/environment.json"] = sha256_lf(ENVIRONMENT)
    manifest = {"schema": "s2-registration-1",
                "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "git_commit": commit, "uncommitted_changes_in_study2_or_kaggle_s2": dirty,
                "hash_convention": "SHA-256 of the file's UTF-8 text with LF line endings (the bytes run on Kaggle)",
                "files_sha256": files,
                "candidates": {cid: {"repo": c.repo, "revision": c.revision, "upstreams": dict(c.upstreams),
                                     "split_rule": c.split_rule} for cid, c in PD.CANDIDATES.items()},
                "candidate_order": list(PD.CANDIDATE_ORDER),
                "kaggle": {"probe": f"{USER}/{PROBE_SLUG}", "prepare": f"{USER}/{PREPARE_SLUG}",
                           "smoke": f"{USER}/{SMOKE_SLUG}", "run": f"{USER}/{RUN_SLUG}", "dataset": S2_DATASET,
                           "pilot_run": PILOT_RUN, "pilot_dataset": PILOT_DATASET,
                           "docker_image": docker_image()}}
    path = REGISTRATION
    path.write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))  # LF: the bytes Kaggle and OTS see
    if dirty:
        print("WARNING: kaggle_s2/ or study2/ has uncommitted changes; commit, then run `freeze` again so the "
              "registration names a commit that contains exactly these files.", file=sys.stderr)
    return path


def build_dataset(prepared: Path, output: Path) -> Path:
    sys.path.insert(0, str(HERE))
    import prepare_domain as PD
    return PD.package(prepared, output, S2_DATASET)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("which", choices=["probe", "prepare", "smoke", "run", "dataset", "pin-image", "freeze"])
    parser.add_argument("--phase", choices=["bootstrap", "gpu"], default="gpu")
    parser.add_argument("--rank", type=int, choices=[0, 1], default=0, help="prepare: 1 = pre-registered fallback")
    parser.add_argument("--session-hours", type=float, default=11.25, help="run: cap to the remaining GPU quota")
    parser.add_argument("--prepared", type=Path, default=ROOT / "data" / "s2-prep" / "s2prep")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "s2-dataset-online")
    args = parser.parse_args()
    if args.which == "probe":
        print(build_probe())
    elif args.which == "prepare":
        print(build_prepare(args.rank))
    elif args.which == "smoke":
        print(build_smoke())
    elif args.which == "run":
        print(build_run(args.phase, args.session_hours))
    elif args.which == "dataset":
        print(build_dataset(args.prepared, args.output))
    elif args.which == "pin-image":
        print(pin_image())
    else:
        print(freeze())
