# Filing Study 2's public registration on OSF

Amendment 1 (`study2/PROTOCOL.md`, Section 15) commits to a public registry entry filed **after** execution,
together with the manifest, the timestamp proof and the amendment. This page lists what to upload and gives the
text to paste. The registration must say plainly that it was filed after execution.

## 1. Files to upload

Upload these files unchanged. Their hashes are checked below.

| File | What it is | SHA-256 |
|---|---|---|
| `study2/PROTOCOL.md` | The frozen protocol, including Amendment 1 | `1c34f675e1352bddf5f6a91e9c44ff77dc8bf50358401d1492fa6a3f6dad15d2` |
| `study2/registration-manifest.json` | The SHA-256 of the protocol and of every Study 2 code file | `3eeda992819ae9428a00c6a98688cf9b172879a136c771d7b9794ed92110e45a` |
| `study2/registration-manifest.json.ots` | OpenTimestamps proof of the manifest (Bitcoin block 970,569) | n/a |
| `study2/timestamp-evidence.json` | Calendar submission time, block details and Kaggle version times | n/a |

Optional: `reports/s2-kaggle/run/s2state/decision-report.txt` (the outcome, clearly labelled as such) and
`paper/main.pdf`.

Before uploading, check that the two hashes still match (PowerShell):

```powershell
Get-FileHash study2\PROTOCOL.md, study2\registration-manifest.json -Algorithm SHA256
```

## 2. Steps

1. On osf.io, create a project, for example "Domain-shift forgetting: TaperNorm", and add the files above.
2. Open **Registrations → New registration**, choose the **Open-Ended Registration** template, and paste the
   summary below.
3. Choose **make public immediately**. An embargo would defeat the purpose here.
4. Once it is registered, add the DOI to the repository README, and to the paper if it is filed before arXiv
   submission.

## 3. Text to paste

**Title:** Study 2: Does TaperNorm increase forgetting where its premise is present? (protocol hash-timestamped
before execution; registered after execution)

**Summary:**

> This registration records the protocol of Study 2, a pre-specified follow-up to a pilot (Study 1) that asked
> whether replacing a pre-norm Transformer's internal RMSNorm with TaperNorm increases forgetting of web text
> after training switches to a new domain.
>
> **Timing (please read).** This entry was filed after Study 2 was executed, as the protocol's Amendment 1
> states. It is not a conventional pre-registration. The plan was fixed before any Study 2 job as follows:
>
> - **The manifest.** The protocol (including Amendment 1) and every Study 2 code file are listed by SHA-256 in
>   `registration-manifest.json` (SHA-256 3eeda992…e45a, git commit 321e783).
> - **Timestamp submission.** The manifest's SHA-256 was submitted to four OpenTimestamps calendars at 21:55 UTC
>   on 8 October 2026, one minute before the first Study 2 job. This time comes from the author's machine.
> - **Embedding in Kaggle versions.** Every Study 2 Kaggle notebook version (created 21:56–22:59 UTC on 8 October)
>   embeds the manifest byte for byte, and Kaggle records their creation times on its servers. The probe,
>   smoke-test and main-run notebooks are public; the data-preparation notebook is private because its output
>   holds the sealed reserved test split.
> - **The Bitcoin attestation.** The `.ots` proof is anchored in Bitcoin block 970,569, mined at 01:59 UTC on
>   9 October. That is about 3 hours after the main run started, and after the Chinese-branch endpoints of seeds
>   101–103 had been computed inside the running session. The Bitcoin anchor alone therefore does not predate
>   every outcome; the earlier bound rests on Kaggle's records.
> - **Runner hash.** Every training session recorded a runner SHA-256 equal to the registered one, and the
>   decision rules were applied in-process by that runner.
>
> **Design (summary).** Paired 17.7M-parameter GPT-style models (internal RMSNorm vs. internal TaperNorm without
> the auxiliary loss; final RMSNorm kept) are trained on 150M web tokens and then continued for 100M tokens on web
> text or on a shifted domain. The endpoint is a difference-in-differences in held-out web cross-entropy.
> - **H2.** A new domain X, chosen by a fixed rule from four candidates, tested on six seeds (101–106).
> - **R2.** A replication of the pilot's web-to-Python contrast on three fresh seeds (104–106).
> - **M.** A pre-specified manipulation check of the activation-scale premise.
>
> The decision rules and the interpretation matrix are in the protocol (Sections 8–9).
>
> **Outcome (for completeness).** X was Chinese web text (mC4 zh). H2: opposite direction (mean D_X −0.0325
> nats/token, 95% interval −0.0683 to +0.0032). R2: stop, small observed effect (mean D +0.0093). Equivalence
> within ±0.015 was not shown in either. Code and all records: https://github.com/Danieldotcomcoder/domain-shift-forgetting
