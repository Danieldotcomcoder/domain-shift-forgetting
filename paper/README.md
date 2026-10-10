# Preprint

`main.tex` is the arXiv preprint covering both studies: Study 1 (the H1 pilot, web → Python) and Study 2 (web →
Chinese, the fresh-seed web → Python replication and the manipulation check). `main.pdf` is the compiled version.

Every number, table and figure is generated from the runs' own records: `reports/h1-kaggle/` (Study 1) and
`reports/s2-kaggle/` plus `study2/` (Study 2). Before writing anything, the script:

- recomputes every contrast, interval and manipulation-check value from the raw evaluation records and checks
  them against the frozen decision reports;
- re-checks the integrity chain: configuration hashes, the registered protocol and runner hashes
  (`study2/registration-manifest.json`), the bit-exact lineage records, and the OpenTimestamps proof if the
  `opentimestamps` package is installed (optional; without it the script prints a note and skips that check).

```bash
python paper/make_assets.py    # writes paper/generated/*.tex, paper/arxiv-abstract.txt and paper/figures/*
```

To check Study 1's configuration hash, logged at 12:19:23 UTC (`c18441799eee…`), yourself: it is not
`sha256sum config.json`, but the hash of the canonical JSON without the two keys added after hashing:

```bash
python -c "import json,hashlib; c=json.load(open('reports/h1-kaggle/h1state/config.json')); [c.pop(k) for k in ('config_sha256','checkpoint_seconds')]; print(hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest())"
```

Study 2's configuration (`reports/s2-kaggle/run/s2state/config.json`) uses the same procedure. To inspect the
timestamp proof of Study 2's registration manifest, use
`python study2/timestamp.py info study2/registration-manifest.json.ots` (needs `opentimestamps`), or `ots verify study2/registration-manifest.json.ots` with the OpenTimestamps client.

Build the PDF with any LaTeX distribution, or upload the `paper/` folder to Overleaf:

```bash
cd paper && tectonic main.tex   # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

For arXiv, submit `main.tex`, `refs.bib`, `generated/`, `figures/trajectories.pdf` and
`figures/study2_trajectories.pdf` as source, and paste `arxiv-abstract.txt` (plain ASCII, generated from the same
numbers as the PDF abstract) into the abstract field.
