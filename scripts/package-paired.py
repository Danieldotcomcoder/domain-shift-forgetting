"""Private paired-worker bundle. No data, checkpoints or credentials included."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

root = Path(__file__).resolve().parents[1]
output = root / "artifacts/kaggle-paired"
output.mkdir(parents=True, exist_ok=True)
paths = list((root / "src").rglob("*.py")) + list((root / "tests").glob("test_*.py"))
paths += [root / "pyproject.toml", root / "docs/paired-kaggle-runbook.md", root / "docs/scientific-runbook.md"]
paths += [root / "configs" / name for name in ("kaggle-stage1-paired.json", "kaggle-stage1.json", "kaggle-stage1-24h-baseline.json", "stage1.v3.json")]
archive = output / "domain-shift-paired.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for path in paths:
        z.write(path, path.relative_to(root).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
shutil.copyfile(root / "notebooks/kaggle-stage1-paired.ipynb", output / "kaggle-stage1-paired.ipynb")
shutil.copyfile(root / "docs/paired-kaggle-runbook.md", output / "paired-kaggle-runbook.md")
receipt = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file() and p.name != "bundle-sha256.json"}
(output / "bundle-sha256.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(f"Verified paired bundle: {archive}")
