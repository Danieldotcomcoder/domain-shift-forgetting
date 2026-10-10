# Study 2 execution log

All times are UTC. GPU quota is Kaggle T4 × 2, billed at 1× elapsed. At 21:54 on 8 October the author reported
18 GPU hours available, with the weekly reset to 30 h about 26 h later (around 00:00 UTC on 10 October).

| Time (UTC) | Step | Notes |
|---|---|---|
| 2026-10-08 ~21:50 | Author decisions | Run now with a hash-timestamp first (Amendment 1); design approved as written (commit `0b3a2ce`) |
| 2026-10-08 21:54 | Docker pin | `pin-image`: `gcr.io/kaggle-private-byod/python@sha256:2757e0c7…` (pilot run notebook) |
| 2026-10-08 21:54 | Freeze | `study2/registration-manifest.json` (commit `fa09196`, files at `321e783`), SHA-256 `3eeda992819ae9428a00c6a98688cf9b172879a136c771d7b9794ed92110e45a` |
| 2026-10-08 21:55 | OpenTimestamps | Digest accepted by 4 calendars; proof `study2/registration-manifest.json.ots` (commit `bc515cd`), pending Bitcoin attestation |
| 2026-10-08 21:56 | Probe pushed | `danny00/s2-domain-probe` version 1 (GPU T4 × 2, internet on). First Study 2 job |
| 2026-10-08 ~22:01 | Probe complete | About 4 min; torch 2.11.0+cu128 (pinned image). Reproduction of the pilot's switch energies: max relative difference **0.0** (gate passed). Selection **mc4-zh**, S = 0.0934 (RMS 0.0757, Taper-minus 0.1111). Ranking zh 0.0934 > ru 0.0893 > de 0.0514 > OpenWebMath 0.0119. Python reference 0.1099. **No candidate exceeds Python** (Protocol 4.6 applies: proceed, with the premise qualification on H2). Records in `reports/s2-kaggle/probe/` |
| 2026-10-08 22:01 | Preparation pushed | `danny00/s2-prepare-domain` version 1 (CPU, internet on); attaches the probe output and the pilot dataset |
| 2026-10-08 22:10 | Preparation complete | 7.6 min, **accepted** (details below). Records in `reports/s2-kaggle/prepare/` |

Preparation details:
- **Sources.** Train shards `c4-zh.tfrecord-00314` and `-00305` of 01024; validation shard `c4-zh-validation.tfrecord-00001` of 00002. All SHA-256-verified against the hub.
- **Raw tokens before deduplication.** Train 126.5M, dev 9.37M, test 9.66M; one filter round.
- **Deduplication.** 698 near-duplicates removed; 0 exact duplicates; 0 duplicates against the frozen pilot corpus. Audit: 0 of 2,443 cross-split pairs and 0 of 2,000 frozen pairs at or above 0.85.
- **Materialized.** X train 110,000,129 tokens (56,276 documents); X dev 2,097,153 (1,139 documents); reserved test 8,388,609 (4,201 documents, sealed).
- **Orders.** The pilot's seed 101–103 orders were reproduced bit for bit (NumPy 2.1.3 on Kaggle).
- **Manifest hashes.** Online manifest `5994e6fb…`; orders manifest `ba2596ce…`.

| Time (UTC) | Step | Notes |
|---|---|---|
| 2026-10-08 22:27 | Download and check | Whole output fetched (752 MB). All 26 online and order files re-hashed against their manifests: no mismatch |
| 2026-10-08 22:30 | Private dataset | `danny00/study2-online-inputs` created: s2online, s2orders, s2upstream; 27 files; no reserved test |
| 2026-10-08 22:30 | Smoke pushed | `danny00/s2-single-notebook-smoke` version 1 (GPU T4 × 2) |
| 2026-10-08 ~22:57 | Smoke **PASSED** | About 26 min; details below. Records in `reports/s2-kaggle/smoke/` |
| 2026-10-08 22:58 | Main run: bootstrap | `danny00/s2-single-notebook-run` version 1 (CPU): wrote `S2-BOOTSTRAP.json` |
| 2026-10-08 22:59 | Main run: session 1 | Version 2 (GPU T4 × 2, pinned image, SESSION_HOURS 11.25). GPU used before it: about 0.5 h of 18 |
| 2026-10-09 10:07 | Session 1 ended | 11.12 h; exit codes 0/0; no restarts; runner SHA-256 `5aa02b3e…` = registered; registration manifest hash logged (details below) |
| 2026-10-09 21:36 | Main run: session 2 | Version 3 (GPU T4 × 2, pinned image, SESSION_HOURS 11.25; about 4.1 h needed). GPU used before it: about 11.7 h of 18 (weekly reset around 00:00 UTC on 10 October) |
| 2026-10-09 21:38 | Timestamp upgraded | The proof now holds a **Bitcoin attestation in block 970,569** (hash `000…1d6cde1`, mined 2026-10-09 01:59:53 UTC). Exact scope below |

Session 1 details:
- **Completed.** Seeds 101–103 (X branch, both conditions) and seeds 104–105 (prefix, web, Python and X).
- **Lineage.** All 30 branch-start checks were bit-exact (max |dCE| 0.0, max relative dE 0.0).
- **Stopped at.** Seed 106 at prefix step 763 in both workers; 26,705 updates remain per worker.

| Time (UTC) | Step | Notes |
|---|---|---|
| 2026-10-10 01:40 | Session 2 ended | 21:37 → 01:40; 4.06 h; exit codes 0/0; no restarts; seed 106 completed in both conditions. All 12 runs complete; 0 missing records; no failures or problems; runner hash = registered |
| 2026-10-10 08:22 | Final report fetched | `reports/s2-kaggle/run/s2state/decision-report.{txt,json}` |
| 2026-10-10 08:27 | All records fetched | All 12 runs' `events.jsonl`, `train.jsonl`, completion receipts and switch receipts, plus the worker logs (114 MB), into `reports/s2-kaggle/run/` |
| 2026-10-10 08:35 | Independent recomputation | `s2_run.py report` was rerun on CPU from the downloaded records and the pilot's committed records. Both decisions, all intervals, all six seeds' evidence, the pooled estimate and the manipulation check are identical. The document bootstrap differs only at about 1e-17 (BLAS rounding, NumPy 2.5.3 vs 2.1.3) |

**GPU use.** Probe about 0.1 h, smoke test 0.45 h, session 1 11.12 h, session 2 4.06 h: about 15.7 h in total.

### Pre-registered results

**H2 (web → mc4-zh, seeds 101–106): OPPOSITE DIRECTION.**
- **Endpoint.** Mean D_X = **−0.0325** (−0.03255) nats/token (threshold ≤ −0.03); SD 0.0341.
  - Per seed (101 → 106): −0.0886, +0.0065, −0.0351, −0.0513, −0.0179, −0.0089.
- **Intervals.** 95% t-interval [−0.0683, +0.0032], which includes 0. 90% t-interval [−0.0606, −0.0045]. Equivalence within ±0.015: no.
- **Same direction elsewhere.** Mean D_X(3050) −0.0338; mean Q −0.0306; mean matched differential forgetting at the 6,104 target −0.0378.
- **Guardrails passed.** Prefix gaps 0.23–0.34%; X adaptation 2.18–2.33 nats.
- **Fresh seeds only (104–106, descriptive).** Mean −0.0260, 95% interval [−0.0816, +0.0296].
- **Document bootstrap (evaluation noise only).** Mean-D 95% [−0.0339, −0.0311]. Seed variation dominates.
- **Premise qualification (Protocol 4.6).** The selected domain's switch-time gap (S 0.0934) did not exceed Python's (0.1099).

**R2 (web → Python, fresh seeds 104–106): STOP — SMALL OBSERVED EFFECT. The pilot's NO replicates.**
- **Endpoint.** Mean D = **+0.0093**; SD 0.0072. Per seed +0.0160, +0.0102, +0.0018.
- **Intervals.** 95% t-interval [−0.0084, +0.0271]. 90% t-interval [−0.0027, +0.0214]. Equivalence within ±0.015: no.
- **Note.** The sign is opposite to the pilot's three seeds, which were all negative.
- **Pooled six seeds (descriptive; includes the pilot's published seeds).** Mean −0.0065, 95% [−0.0254, +0.0125].

**Manipulation check (descriptive).** Values are RMS / Taper-minus, X vs Python.
- **Switch gap:** 0.0855 / 0.1217 for X vs 0.0926 / 0.1361 for Python. Not larger for X.
- **Domain-specific change in training:** 0.1652 / 0.1830 for X vs 0.0386 / 0.0418 for Python. About 4× larger for X.
- **Reading:** "premise stronger for X in training" = **yes**; "at the switch" = no.
- **Interpretation (matrix in Protocol 9.4):** opposite direction with the in-training premise stronger for X is evidence against the scale-mismatch account of H1 in this setting.

What the timestamp evidence establishes:
- **Bitcoin.** The block proves that the manifest (protocol and code hashes) existed by 01:59 UTC on 9 October (Bitcoin block times are accurate only to about two hours). This is *after* the X branches of seeds 101-103 reached their endpoints inside the running session (worker logs: 23:59, 00:57-00:58 and 01:55-01:56 UTC). *Corrected 10 October:* an earlier version of this line said that no seed had completed by then, which was wrong. The Bitcoin anchor alone therefore does not predate every outcome; the two items below do.
- **Before the first job.** The digest was submitted to the calendars at 21:55 UTC on 8 October, before the first Study 2 job at 21:56. The calendars' receipts are in the proof, but that earlier time is not Bitcoin-anchored.
- **Kaggle.** Every Study 2 notebook version embeds the manifest byte for byte: the probe at 21:56, the preparation at 22:01, the smoke test at 22:30, and the main run at 22:58 and 22:59 UTC on 8 October. The probe's selection, the preparation manifest and every runner session record its SHA-256. These are server-side timestamps, verifiable once the notebooks are made public.

Smoke test details:
- **(A) Mini protocol.** Complete; no failures or problems; 24/24 lineage checks passed; resume across 3 sessions; a re-run does nothing. The mini decisions come from 20-update toy models and are meaningless.
- **(B) Lineage of the six real pilot switch states.** Bit-exact: CE and all 24 per-site energies differ by 0.0.
- **(C) Real protocol, seed 104 prefix only.** 0.381 s/update (RMS); 0.377 s during calibration and 0.391 s after it (Taper-minus); full-dev evaluation about 19–21 s per domain; no FP16 retries; resume works.
