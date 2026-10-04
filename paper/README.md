# Preprint draft

`main.tex` is the first draft of the arXiv preprint for the H1 pilot. `main.pdf` is the compiled draft.
Red **[Author: …]** notes in the text are facts only the author can confirm (name, affiliation, protocol
date, AI-assistance wording); resolve them before submission.

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
