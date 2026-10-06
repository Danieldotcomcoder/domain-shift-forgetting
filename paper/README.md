# Preprint draft

`main.tex` is the draft arXiv preprint for the H1 pilot. `main.pdf` is the compiled draft.

Every number, table and figure is generated from the run's own records in `reports/h1-kaggle/`. The
script recomputes the primary contrasts from the raw evaluation logs and checks them against the frozen
decision report before writing anything:

```bash
python paper/make_assets.py    # writes paper/generated/*.tex and paper/figures/trajectories.{pdf,png}
```

Build the PDF with any LaTeX distribution, or upload the `paper/` folder to Overleaf:

```bash
cd paper && tectonic main.tex   # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

For arXiv, submit `main.tex`, `refs.bib`, `generated/` and `figures/trajectories.pdf` as source.
