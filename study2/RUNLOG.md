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

What the timestamp evidence establishes:
- **Bitcoin.** The block proves that the manifest (protocol and code hashes) existed by 01:59 UTC on 9 October. At that time session 1 had completed no seed: S102's X branches finished at 00:57 and 00:58, but nobody saw results before the session's output appeared at 10:07.
- **Before the first job.** The digest was submitted to the calendars at 21:55 UTC on 8 October, before the first Study 2 job at 21:56. The calendars' receipts are in the proof, but that earlier time is not Bitcoin-anchored.
- **Kaggle.** Every Study 2 notebook version embeds the manifest byte for byte: the probe at 21:56, the preparation at 22:01, the smoke test at 22:30, and the main run at 22:58 and 22:59 UTC on 8 October. The probe's selection, the preparation manifest and every runner session record its SHA-256. These are server-side timestamps, verifiable once the notebooks are made public.

Smoke test details:
- **(A) Mini protocol.** Complete; no failures or problems; 24/24 lineage checks passed; resume across 3 sessions; a re-run does nothing. The mini decisions come from 20-update toy models and are meaningless.
- **(B) Lineage of the six real pilot switch states.** Bit-exact: CE and all 24 per-site energies differ by 0.0.
- **(C) Real protocol, seed 104 prefix only.** 0.381 s/update (RMS); 0.377 s during calibration and 0.391 s after it (Taper-minus); full-dev evaluation about 19–21 s per domain; no FP16 retries; resume works.
