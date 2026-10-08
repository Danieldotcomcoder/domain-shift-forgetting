"""Push, check, fetch or continue the Study 2 Kaggle notebooks and dataset. Kaggle API calls only; no local compute.

  .venv\\Scripts\\python kaggle_s2\\s2_status.py probe --push         # start the built notebook on Kaggle
  .venv\\Scripts\\python kaggle_s2\\s2_status.py probe                # status; fetch the selection
  .venv\\Scripts\\python kaggle_s2\\s2_status.py prepare [--download] # status; --download fetches the whole prepared
                                                                     # output to data/s2-prep/ for packaging
  .venv\\Scripts\\python kaggle_s2\\s2_status.py dataset [--create]   # status; --create uploads data/s2-dataset-online
                                                                     # as the PRIVATE dataset
  .venv\\Scripts\\python kaggle_s2\\s2_status.py smoke                # status; fetch smoke-results.json
  .venv\\Scripts\\python kaggle_s2\\s2_status.py run [--continue]     # status; decisions when available; --continue
                                                                     # starts the next session if unfinished
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
USER = "danny00"
KERNELS = {"probe": f"{USER}/s2-domain-probe", "prepare": f"{USER}/s2-prepare-domain",
           "smoke": f"{USER}/s2-single-notebook-smoke", "run": f"{USER}/s2-single-notebook-run"}
FOLDERS = {"probe": "kernel-probe", "prepare": "kernel-prepare", "smoke": "kernel-smoke", "run": "kernel-run"}
DATASET = f"{USER}/study2-online-inputs"
DATASET_FOLDER = ROOT / "data" / "s2-dataset-online"
PATTERNS = {"probe": r"(selection\.(txt|json)|probe-windows/manifest\.json)$",
            "prepare": r"(acceptance\.json|s2online/(manifest|audit|selection)\.json|s2orders/manifest\.json)$",
            "smoke": r"(smoke-results\.json|decision-report\.(txt|json)|sessions\.jsonl)$",
            "run": r"(decision-report\.(txt|json)|sessions\.jsonl|config\.json|S2-STATE\.json)$"}
OUT = ROOT / "reports" / "s2-kaggle"


def kaggle(*args: str) -> str:
    env = dict(os.environ, PYTHONUTF8="1")
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            key, value = line.split("=", 1)
            env.setdefault(key.strip(), value.strip())
    result = subprocess.run([sys.executable, "-X", "utf8", "-m", "kaggle.cli", *args], capture_output=True,
                            text=True, encoding="utf-8", cwd=ROOT, env=env)
    return result.stdout + result.stderr


def registered_runner_sha256() -> str | None:
    path = ROOT / "study2" / "registration-manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["files_sha256"].get("kaggle_s2/s2_run.py")


def push(stage: str) -> None:
    folder = ROOT / "kaggle_s2" / FOLDERS[stage]
    if not (folder / "kernel-metadata.json").exists():
        raise SystemExit(f"Build it first: python kaggle_s2/build_notebooks.py {stage}")
    print(kaggle("kernels", "push", "-p", str(folder)))
    print("Pushed. Follow it at https://www.kaggle.com/code/" + KERNELS[stage])


def dataset(create: bool) -> None:
    if create:
        if not (DATASET_FOLDER / "dataset-metadata.json").exists():
            raise SystemExit("Build it first: python kaggle_s2/build_notebooks.py dataset")
        if any(DATASET_FOLDER.rglob("x_test*")):
            raise SystemExit("The reserved test must never be uploaded to the training dataset")
        print(kaggle("datasets", "create", "-p", str(DATASET_FOLDER), "--dir-mode", "zip"))  # private by default
    print(kaggle("datasets", "status", DATASET))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=sorted(KERNELS) + ["dataset"])
    parser.add_argument("--push", action="store_true", help="push the built notebook (starts it on Kaggle)")
    parser.add_argument("--continue", dest="resume", action="store_true")
    parser.add_argument("--download", action="store_true", help="prepare: fetch the complete output")
    parser.add_argument("--create", action="store_true", help="dataset: upload data/s2-dataset-online (private)")
    args = parser.parse_args()
    if args.stage == "dataset":
        dataset(args.create)
        return
    if args.push:
        push(args.stage)
        return
    kernel = KERNELS[args.stage]
    status = kaggle("kernels", "status", kernel).strip()
    print(status)
    if "RUNNING" in status or "QUEUED" in status:
        print("Still running on Kaggle. Live log: https://www.kaggle.com/code/" + kernel)
        return
    out = OUT / args.stage
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare" and args.download:
        target = ROOT / "data" / "s2-prep"
        print(kaggle("kernels", "output", kernel, "-p", str(target), "-o").strip()[-800:])
        print(f"Prepared output in {target}. Next: python kaggle_s2/build_notebooks.py dataset")
        return
    print(kaggle("kernels", "output", kernel, "-p", str(out), "-o", "--file-pattern", PATTERNS[args.stage]).strip()[-800:])
    if args.stage == "probe":
        for path in sorted(out.rglob("selection.txt")):
            print("\n" + path.read_text(encoding="utf-8"))
        return
    if args.stage in ("prepare", "smoke"):
        for name in ("acceptance.json", "smoke-results.json"):
            for path in sorted(out.rglob(name)):
                print("\n" + path.read_text(encoding="utf-8")[:6000])
        return
    reports = sorted(out.rglob("decision-report.txt"))
    finished = False
    if reports:
        print("\n" + reports[0].read_text(encoding="utf-8"))
        report = json.loads(reports[0].with_suffix(".json").read_text(encoding="utf-8"))
        finished = report["complete"] or bool(report["failures"])
    else:
        print("No report in the latest output yet.")
    registered = registered_runner_sha256()
    for path in sorted(out.rglob("sessions.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            session = json.loads(line)
            match = "matches" if session["code_sha256"] == registered else "DIFFERS FROM" if registered else "vs (no)"
            print(f"Session {session['session']}: {session.get('hours', 0):.2f} h, torch {session['torch']}, runner "
                  f"{session['code_sha256'][:12]} {match} the registered runner hash")
    if finished:
        print("Experiment finished. Full report: " + str(reports[0].with_suffix(".json")))
    elif args.resume:
        push("run")
        print("Next session started; it resumes from the previous output automatically.")
    else:
        print("Not finished. Run again with --continue to start the next session.")


if __name__ == "__main__":
    main()
