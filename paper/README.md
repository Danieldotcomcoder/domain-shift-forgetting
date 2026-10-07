# Preprint draft

`main.tex` is the draft arXiv preprint for the H1 pilot. `main.pdf` is the compiled draft.

Every number, table and figure is generated from the run's own records in `reports/h1-kaggle/`. The
script recomputes the primary contrasts from the raw evaluation logs and checks them against the frozen
decision report before writing anything:

```bash
python paper/make_assets.py    # writes paper/generated/*.tex and paper/figures/trajectories.{pdf,png}
```

To check the configuration hash the run logged at 12:19:23 UTC (`c18441799eee…`) yourself — note that it is not
`sha256sum config.json`, but the hash of the canonical JSON without the two keys added after hashing:

```bash
python -c "import json,hashlib; c=json.load(open('reports/h1-kaggle/h1state/config.json')); [c.pop(k) for k in ('config_sha256','checkpoint_seconds')]; print(hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest())"
```

Build the PDF with any LaTeX distribution, or upload the `paper/` folder to Overleaf:

```bash
cd paper && tectonic main.tex   # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

For arXiv, submit `main.tex`, `refs.bib`, `generated/` and `figures/trajectories.pdf` as source, and paste
`arxiv-abstract.txt` (plain ASCII, generated from the same numbers as the PDF abstract) into the abstract field.
