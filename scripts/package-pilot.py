"""Small original-protocol code bundle; deliberately excludes data and credentials."""
from pathlib import Path
import zipfile
import shutil
import hashlib
import json

root = Path(__file__).resolve().parents[1]
output = root / "artifacts/kaggle-stage1"
output.mkdir(parents=True, exist_ok=True)
paths = list((root / "src").rglob("*.py")) + list((root / "tests").glob("test_*.py"))
paths += [root / "pyproject.toml", root / "configs/kaggle-stage1.json", root / "configs/stage1.v3.json", root / "configs/kaggle-stage1-24h-baseline.json",
          root / "docs/scientific-runbook.md"]
with zipfile.ZipFile(output / "domain-shift-stage1.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
    for path in paths:
        bundle.write(path, path.relative_to(root).as_posix())
print(output / "domain-shift-stage1.zip")

for name in ("kaggle-stage1-prepare.ipynb", "kaggle-stage1-run.ipynb"):
    shutil.copyfile(root / "notebooks" / name, output / name)
shutil.copyfile(root / "docs/scientific-runbook.md", output / "scientific-runbook.md")
receipt = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file() and p.name != "bundle-sha256.json"}
(output / "bundle-sha256.json").write_text(json.dumps(receipt, indent=2))
with zipfile.ZipFile(output / "domain-shift-stage1.zip") as bundle:
    assert bundle.testzip() is None
print("Bundle verified; companion notebooks and runbook copied.")
