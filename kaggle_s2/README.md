# Study 2 on Kaggle: runbook

The scientific source of truth is [`study2/PROTOCOL.md`](../study2/PROTOCOL.md). This page covers only how to run
it. Every compute step runs on Kaggle; nothing trains on the local machine. Commands are for PowerShell from the
repository root. The Kaggle token is read from `.env` (`KAGGLE_API_TOKEN=...`), as for the pilot.

| File | Role |
|---|---|
| `s2_run.py` | GPU runner: verifies the pilot assets and all data, trains, evaluates, runs the lineage checks, and applies the H2/R2 decision rules (copied pilot code + Study 2 jobs and analyses) |
| `probe_domains.py` | Domain-selection probe: forward passes on the pilot's six switch states, then `selection.json` |
| `prepare_domain.py` | CPU preparation of the selected domain: streaming, tokenization, dedup, packing, orders, acceptance checks, packaging |
| `build_notebooks.py` | Builds the four notebooks and the dataset folder, pins the Docker image, writes the registration manifest |
| `s2_status.py` | Pushes, checks and fetches notebooks; uploads the private dataset; continues the main run |
| `test_s2_run.py` | Light CPU tests (about 35 s): equivalence with the pilot's runner, the pilot pins, a synthetic end-to-end decision report, the prep-to-runner contract, and the worker and orchestrator with training stubbed out |

**Never modify `kaggle_h1/h1_run.py`.** The paper cites its hash, and the smoke test embeds it unchanged.

## 0. Before registration (no Kaggle job runs)

```powershell
.venv\Scripts\python -m pytest kaggle_s2\test_s2_run.py -q    # light CPU tests; never touches the GPU
.venv\Scripts\python kaggle_s2\build_notebooks.py pin-image   # reads the pilot notebook's Docker image (metadata only)
git add kaggle_s2 study2; git commit -m "Study 2: protocol and code for registration"
.venv\Scripts\python kaggle_s2\build_notebooks.py freeze      # writes study2/registration-manifest.json
git add study2/registration-manifest.json; git commit -m "Study 2: registration manifest"
```

- `pin-image` records the image in `kaggle_s2/environment.json`, which `freeze` then hashes. If the pilot's
  image cannot be read, GPU notebooks use Kaggle's current image (Protocol 11), and the deviation is recorded.
- Timestamp the manifest before any Study 2 notebook is pushed (Protocol Amendment 1). Only its SHA-256 is sent:
  `.venv\Scripts\ots stamp study2\registration-manifest.json`. Commit the `.ots` proof. A few hours later, run
  `.venv\Scripts\ots upgrade study2\registration-manifest.json.ots` to complete the Bitcoin attestation, and commit
  again.
- Every notebook embeds the manifest byte for byte, and the builder refuses to build without it.
- File the public registration (for example on OSF) with `study2/PROTOCOL.md`, the manifest, the `.ots` proof and
  the commit.

## 1. Domain-selection probe (GPU T4, internet on, ~0.3 h)

```powershell
.venv\Scripts\python kaggle_s2\build_notebooks.py probe
.venv\Scripts\python kaggle_s2\s2_status.py probe --push
.venv\Scripts\python kaggle_s2\s2_status.py probe             # when finished: prints selection.txt
```

- Inputs: the pilot dataset `danny00/stage1-online-inputes` and the pilot run notebook `danny00/h1-single-notebook-run`
  (its output holds the six switch states; never delete it).
- Expected output: `SELECTED: <id> (selected)` and the reproduction line `... passed`.
- If the output is `NO SELECTION: reproduction_failed`, stop. A gate failed (Protocol 4.5), and this is not a
  result.

## 2. Prepare the selected domain (CPU only, internet on, ~1–3 h)

```powershell
.venv\Scripts\python kaggle_s2\build_notebooks.py prepare
.venv\Scripts\python kaggle_s2\s2_status.py prepare --push
.venv\Scripts\python kaggle_s2\s2_status.py prepare           # acceptance.json: "status": "accepted"
```

- The notebook attaches the probe notebook's output, so the selection is read as data, never typed by hand.
- If preparation reports that quotas cannot be met, Protocol 5.9 applies: rank-1 fallback, recorded as a deviation.
  Rebuild with `build_notebooks.py prepare --rank 1` and push again. No other rank is accepted.

## 3. Private dataset (no reserved test)

```powershell
.venv\Scripts\python kaggle_s2\s2_status.py prepare --download    # whole output -> data\s2-prep\ (git-ignored)
.venv\Scripts\python kaggle_s2\build_notebooks.py dataset         # data\s2-dataset-online: s2online, s2orders, s2upstream
.venv\Scripts\python kaggle_s2\s2_status.py dataset --create      # uploads it as PRIVATE danny00/study2-online-inputs
```

`data\s2-prep\s2prep\reserved-test\` is the sealed X test set. Keep it out of every notebook, and back it up
privately. `package` refuses to copy it.

## 4. Smoke test (GPU T4 x2, ~0.75 h; separate notebook, no outcome data)

```powershell
.venv\Scripts\python kaggle_s2\build_notebooks.py smoke
.venv\Scripts\python kaggle_s2\s2_status.py smoke --push
.venv\Scripts\python kaggle_s2\s2_status.py smoke             # smoke-results.json; the log ends "SMOKE PASSED"
```

The smoke test checks three things:
- (A) the mini protocol end to end on a mini "pilot" made by the pilot's own runner;
- (B) the lineage of the six real switch states. Expect `max_ce_diff` 0.0 when the Docker image is pinned;
- (C) throughput and resume on seed 104's web prefix only.

If (B) fails, do not start the main run.

## 5. Main run (GPU T4 x2, ~15.5 h in two sessions)

```powershell
.venv\Scripts\python kaggle_s2\build_notebooks.py run --phase bootstrap
.venv\Scripts\python kaggle_s2\s2_status.py run --push          # CPU, seconds: authorizes one fresh start
.venv\Scripts\python kaggle_s2\build_notebooks.py run --phase gpu
.venv\Scripts\python kaggle_s2\s2_status.py run --push          # session 1 (stops itself at 11.25 h)
.venv\Scripts\python kaggle_s2\s2_status.py run --continue      # after it ends: session 2, resumes automatically
.venv\Scripts\python kaggle_s2\s2_status.py run                 # decision-report.txt: H2 and R2 answers
```

- **Order.** Each worker runs the X branches of 101, 102 and 103, then 104 → 106 (prefix → web → Python → X).
- **Failures.** A numerical or lineage failure stops both workers and is recorded. Investigate it, and never replace
  a seed.
- **State.** All state lives in the notebook's own output (`s2state/`). Re-running a finished experiment does nothing.

## 6. After the run

`s2_status.py run` downloads the decision report, configuration and session record into `reports\s2-kaggle\run\`.
For the paper, also fetch every `events.jsonl` and `train.jsonl` with
`kaggle kernels output danny00/s2-single-notebook-run --file-pattern "(events|train)\.jsonl$"`. Commit the
generated `kaggle_s2\kernel-*` folders with each push, so that the repository holds exactly what ran.
