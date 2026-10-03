"""Build a small upload bundle without datasets, environments, or credentials."""
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
output = root / "artifacts" / "kaggle"
output.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(output / "domain-shift-kaggle.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
    files = list((root / "src").rglob("*.py")) + [root / "pyproject.toml", root / "docs/kaggle.md"]
    for path in files:
        bundle.write(path, path.relative_to(root).as_posix())
print(output / "domain-shift-kaggle.zip")
