# Study 2 — handoff for a new session

Start here if you are a new Claude Code session continuing this project. Read this whole file before doing
anything. Branch: `study2` (created from `main` at `4de359f`, the reviewer-verified paper commit).

## Status (updated 2026-10-08): protocol and all code written and CPU-tested; NOTHING has run on Kaggle/GPU

- `study2/PROTOCOL.md` is the pre-registration document. It is complete and consistent with the code; a test checks
  that it quotes every pin and constant.
- `kaggle_s2/` holds all the code:
  - `s2_run.py`: the GPU runner;
  - `probe_domains.py`: the selection probe;
  - `prepare_domain.py`: the X data preparation;
  - `build_notebooks.py`: the notebooks, private dataset folder, Docker pin and registration manifest;
  - `s2_status.py`: push, status, fetch, continue and dataset upload;
  - `test_s2_run.py`: 71 light CPU tests, all passing in about 35 s; 1 opt-in heavy test via `S2_HEAVY_TESTS=1`;
  - `README.md`: the step-by-step runbook.
- Not committed yet: everything under `study2/` and `kaggle_s2/` is untracked on branch `study2`.
- Not done, on purpose (user's call):
  - `build_notebooks.py pin-image` (a Kaggle metadata read);
  - `freeze` (needs a commit first);
  - the OSF registration;
  - any Kaggle push.
- Next steps: the runbook's step 0 (kaggle_s2/README.md), then registration, then steps 1–6. No notebook may be
  pushed before the registration timestamp.

### Design decisions made while formalizing (review before registering; all are in PROTOCOL.md)

1. Candidates: the four proposed below, with pinned revisions verified on the HF API: `allenai/c4@1588ec45…` (the
   pilot's own revision) for mc4-de/ru/zh, and `open-web-math@fde8ef8d…`. All are ODC-BY + Common Crawl ToU. The
   `-Latn` variants are excluded by the file patterns.
2. Selection rule = Finding A's switch gap, averaged over 3 seeds × 2 conditions × 12 sites. Python reference
   S = 0.1099. The probe must first reproduce the pilot's recorded energies (rel 1e-3), else no selection.
   Probe sample = the first 256×512+1 train-side tokens of the first shard in the pilot's shard order.
3. If no candidate beats Python's gap, Study 2 still proceeds; H2 is read with a premise qualification
   (Protocol 4.6, 9.4).
4. Pre-specified fallback if X cannot meet its quotas: the rank-1 candidate. Only ranks 0 and 1 are accepted by
   the runner.
5. NEW vs pilot: a secondary equivalence reading (TOST, 90% t-interval within ±0.015). The pilot said "STOP is not
   equivalence"; this lets Study 2 say more. It is removable if the user prefers pure screening.
6. NEW: a lineage gate. At s=0 of every branch the restored switch state must re-evaluate to its own records
   (1e-3 nats; rel 1e-3 energies). The pilot's records show bit-identical re-evaluations. GPU notebooks pin the
   pilot's Docker image.
7. X data:
   - mC4 uses the official train/validation files with the pilot's C4 50/50 dev/test rule.
   - OpenWebMath uses URL-group SHA-256 mod 100 (90/5/5).
   - Dedup: exact and near (pilot settings); the frozen pilot corpus always wins. Near-dedup vs pilot TRAINING
     arrays is deliberately not done (Protocol 5.4).
8. The X branch evaluates web and X dev (not Python dev). Diagnostics probe web, Python and X everywhere. The X
   adaptation baseline is the X branch's s=0.
9. Orders:
   - seeds 104–106 web/Python orders use the pilot generator (verified to reproduce seeds 101–103 bit for bit);
   - X orders: `default_rng([20261008, seed])`;
   - bootstrap seed 20261011.
10. The smoke test produces no outcome data: (A) mini protocol on a mini "pilot" made by h1_run.py; (B) lineage of
    the real switch states; (C) seed 104 prefix only.
11. Job order per GPU: the X branches of 101–103 first, then 104–106 (prefix → web → Python → X). The total is
    3.30B tokens, about 15.5 h per GPU (2 sessions) plus about 1 h for the probe and smoke test.

## What the user asked for

1. Write the **Study 2 protocol** (to be publicly pre-registered by the user, e.g. on OSF, before any GPU use).
2. Write **all Kaggle code** for Study 2 (data preparation, domain-selection probe, training/evaluation runner,
   notebook builder, status script, tests).
3. **Do not run anything** — no Kaggle pushes, no GPU, no local training. The user will approve runs later.
4. Standing rule: never run training or GPU work on the user's PC (it freezes); light CPU checks only.

## Where things stand (Study 1 = the H1 pilot, finished)

- H1 pilot (protocol v3): RMS vs Taper-minus (internal TaperNorm, no aux loss, final RMSNorm kept), 6-layer
  17.7M-param GPT-2-vocab LM, seeds 101–103, web prefix 9,156 updates → web branch and Python branch of 6,104
  updates each. Primary endpoint D at s = 6,104. Result: mean D = −0.0223 (−0.0200, −0.0207, −0.0262) →
  pre-specified decision **STOP — SMALL OBSERVED EFFECT** (H1 = No). One 9.36 h Kaggle session, 3 Oct 2026.
- Runner: `kaggle_h1/h1_run.py` (SHA-256 `66769e7d…`). **Never modify this file** — the paper cites its hash
  at every commit and `paper/make_assets.py` asserts it. Study 2 code goes in new files.
- Records: `reports/h1-kaggle/` (decision report, config, session log, per-run `events.jsonl`, `train.jsonl`,
  completion receipts). Paper: `paper/` (LaTeX; every number generated by `paper/make_assets.py`).
- Paper status: an external AI reviewer judged it ready for arXiv (overall 7; significance 4 because of scope:
  one small model, one domain pair, three seeds). Study 2 exists to address that for TMLR.
- **Pending on `main`, not on this branch** (do them on main if the user asks): round §5.3 early-window values
  consistently (−0.018 vs −0.0185 → use 4 decimals in both); make Table 6 float appear after the §5.4 heading;
  the user decides whether the affiliation stays "University of the People" or becomes "Independent
  Researcher"; tag the final paper commit `paper-v1` after those fixes.

## Agreed design direction for Study 2 (from the conversation)

- A **separate, publicly pre-registered study** — not an extension of the pilot. Register before any GPU run.
- **Why:** the pilot's exploratory "Finding A" showed H1's premise was weak for web→Python: the code-specific
  change in activation scale at the 12 internal normalizer inputs was only ~0.042 in log scale (~4%) in both
  architectures, and RMS's large scale shrinkage (×0.81) happened equally in the web control branch. Study 2
  should test H1 **where the premise is present**, chosen by a pre-registered rule.
- **Components (target ~17 h of T4×2 time):**
  1. Domain-selection probe (forward passes only, < 0.5 h): on the saved switch states of seeds 101–103,
     measure the switch-time scale gap between each candidate domain and web text.
  2. New-domain branch for seeds 101–103 from the existing switch states (~3.5 h). The web-branch control
     already exists in the pilot records.
  3. Three fresh seeds 104–106, each a full prefix plus web, Python and new-domain branches (~12 h). This also
     replicates web→Python on independent seeds.
  4. Smoke test on a separate notebook slug (~0.5 h).
- **Do not:** add seeds to the pilot's analysis (protocol forbids outcome-driven extra seeds), or use another
  programming language as the new domain (likely reproduces the weak premise).

## Candidate domains (proposed to the user)

All are web-scale text, freely licensed, not gated, streamable with a pinnable revision. Verify file layouts and
licenses at the pinned revisions before freezing the protocol.

| Candidate | Source (verify) | Kind of shift |
|---|---|---|
| German web text | mC4 `de`, hosted in `allenai/c4` (multilingual) | Same script, different language |
| Russian web text | mC4 `ru` (`allenai/c4` multilingual) | Different script (Cyrillic); mostly byte-level GPT-2 tokens rare in the English prefix |
| Chinese web text | mC4 `zh` (`allenai/c4` multilingual) | Different script; extreme token-distribution shift |
| Mathematical web text | `open-web-math/open-web-math` | LaTeX-heavy; Latin tokens shared with web, different statistics |

**Selection rule (draft, to be fixed in the protocol before probing):** pick the candidate with the largest
mean absolute natural-log scale ratio (candidate vs web) at the 12 internal normalizer inputs at the switch,
averaged over seeds 101–103 and both conditions — exactly Finding A's "code vs web at the switch" measure
(pilot values: 0.097 RMS, 0.123 Taper-minus for Python). Report all candidates. Known limitation: this is a
proxy; Finding A's key quantity was the scale *change during training*, which a forward probe cannot see.
Pre-register what happens if no candidate exceeds Python's gap.

## Draft hypotheses and analysis (to formalize)

- **Primary (H2):** web→X (selected domain), seeds 101–106 (6 seeds): D_X at s = 6,104 with the pilot's
  estimand, guardrails and thresholds (0.015 / 0.03 investment thresholds; note the protocol never derived
  them). With 6 seeds the descriptive t-interval multiplier is t₀.₉₇₅,₅ = 2.571. Disclose that seeds 101–103
  reuse pilot prefixes whose prefix-level results are already public.
- **Secondary (R2):** web→Python replication on seeds 104–106 (independent), plus a clearly labeled pooled
  6-seed estimate.
- **Pre-specified manipulation check:** the Finding A measures (switch gap, branch-wise web-probe scale change,
  domain-specific change) for X vs Python.
- Keep: matched adaptation, per-document and token-class decomposition, R flag, transient rule, document
  bootstrap (descriptive), per-phase clipping.

## Engineering plan (code only; nothing is run)

New files under `kaggle_s2/` (suggested):
- `prepare_domain.py` — CPU Kaggle kernel: stream the selected domain from Hugging Face at a pinned revision,
  GPT-2 tokenize (tiktoken `gpt2`, EOS per document), exact + MinHash/LSH dedup (same settings as the pilot:
  5-gram shingles, 128 permutations, 32×4 bands, Jaccard ≥ 0.85), document-hash split into train/dev/test,
  materialize `x_train.bin`, `x_dev.bin` (+ owners, documents) and a sealed test set; generate orders for seeds
  104–106 (web, python, X) and X orders for 101–103. Reuse logic from `src/domain_shift_forgetting/pilot_data.py`
  and `pilot_arrays.py` (`make_orders`). Quotas mirror the pilot: ≥ 110M X training tokens (the branch needs
  6,104 × 32 windows = 100.0M tokens), 2,097,152 dev labels, 8,388,608 reserved-test labels.
- `probe_domains.py` — short GPU kernel: load `switch.pt` for seeds 101–103 × both conditions from the pilot
  notebook output, stream ~131,072 tokens per candidate, compute the gap, write the selection.
- `s2_run.py` — GPU runner generalizing `kaggle_h1/h1_run.py` (copy the model/optimizer/eval/diagnostic code;
  do not import-modify the original): jobs = X branch from existing switch states (seeds 101–103) + full runs
  for seeds 104–106 with three branches; same evaluation schedule; same self-resuming notebook pattern (kernel
  lists its own output as input; bootstrap marker authorizes one fresh start); a frozen config hash logged at
  start; decision rules applied in-process at the end.
- `build_notebooks.py`, `s2_status.py`, `test_s2_run.py` — mirror the `kaggle_h1/` versions.

## Key facts and locations

- Kaggle user `danny00`; API token in `.env` as `KAGGLE_API_TOKEN` (use
  `.venv/Scripts/python -X utf8 -m kaggle.cli …`; on Windows set `PYTHONUTF8=1`).
- Prepared pilot corpus: Kaggle dataset `danny00/stage1-online-inputes` (`online/` arrays, `orders/` for seeds
  101–103, `upstream/`). Local copy (git-ignored): `data/kaggle-online/`.
- Pilot run notebook: `danny00/h1-single-notebook-run`. **Its output holds the switch states (`h1state/runs/
  S{seed}-{condition}/switch.pt`, ~1.3 GB) that Study 2 reuses — never delete it.** Mount path when attached
  as a kernel source: `/kaggle/input/h1-single-notebook-run/h1state/…`.
- Kaggle quota: 30 h/week of T4×2, billed at 1× elapsed (the 9.36 h pilot used exactly 9.36 h). Resets
  2026-10-10 00:00 UTC. Commit sessions are capped at 12 h; the pilot runner stops itself at 11.25 h.
- Measured throughput (pilot): ~0.37 s/update per T4 (RMS ≈ Taper-minus), full-dev eval ~37 s, quick ~4.6 s.
- Kaggle behaviour verified: a kernel listing itself in `kernel_sources` receives its latest version's output
  at `/kaggle/input/<slug>/`; dataset inputs mount under `/kaggle/input/` (sometimes nested under
  `datasets/<user>/`), so locate files with `rglob`; `kernels output <slug>/<version>` ignores the version.
- LaTeX: no local TeX install; the previous session used Tectonic from a temp scratch directory (may be gone).

## Open questions for the user

1. Approve or change the four candidate domains, and the design decisions listed under "Status" above. In
   particular, decide whether to keep the new equivalence reading (#5).
2. OSF (or other registry) account for the public registration; optionally make the Study 2 Kaggle notebooks
   public at registration time for a server-side timestamp.
3. The AI-assistance note in the protocol header (a placeholder for the author).
4. Paper polish items and affiliation (see "Pending on main" above).

The original planning notes above (from "Agreed design direction" on) are kept for context. Where they differ from PROTOCOL.md, PROTOCOL.md wins.
