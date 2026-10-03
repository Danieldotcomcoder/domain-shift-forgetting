"""Check, continue, or fetch the H1 Kaggle run. Only Kaggle API calls; no local compute.

  .venv\\Scripts\\python kaggle_h1\\h1_status.py              status; prints the H1 answer when available
  .venv\\Scripts\\python kaggle_h1\\h1_status.py --continue   start the next Kaggle session if the last one
                                                            ended unfinished (it resumes automatically)
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
KERNEL = "danny00/h1-single-notebook-run"
OUT = ROOT / "reports" / "h1-kaggle"


def kaggle(*args: str) -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
    result = subprocess.run([sys.executable, "-X", "utf8", "-m", "kaggle.cli", *args], capture_output=True,
                            text=True, encoding="utf-8", cwd=ROOT)
    return result.stdout + result.stderr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--continue", dest="resume", action="store_true")
    args = parser.parse_args()
    status = kaggle("kernels", "status", KERNEL).strip()
    print(status)
    if "RUNNING" in status or "QUEUED" in status:
        print("Still running on Kaggle. Live log: https://www.kaggle.com/code/" + KERNEL)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", KERNEL, "-p", str(OUT), "-o", "--file-pattern",
                 r"(decision-report\.(txt|json)|sessions\.jsonl|config\.json|H1-STATE\.json)$").strip()[-500:])
    reports = sorted(OUT.rglob("decision-report.txt"))
    if reports:
        print("\n" + reports[0].read_text(encoding="utf-8"))
        report = json.loads(reports[0].with_suffix(".json").read_text(encoding="utf-8"))
        finished = report["complete"] or report["failures"]
    else:
        print("No report in the latest output yet.")
        finished = False
    if finished:
        print("Experiment finished. Full report: " + str(reports[0].with_suffix(".json")))
    elif args.resume:
        print(kaggle("kernels", "push", "-p", str(ROOT / "kaggle_h1" / "kernel-run")))
        print("Next session started; it resumes from the previous output automatically.")
    else:
        print("Not finished. Run again with --continue to start the next session.")


if __name__ == "__main__":
    main()
