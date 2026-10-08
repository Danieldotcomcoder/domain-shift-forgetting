#!/usr/bin/env python3
"""Study 2 domain-selection probe (GPU, forward passes only; see study2/PROTOCOL.md, Sec. 4).

For every pilot switch state (seeds 101-103 x RMS/Taper-minus) it measures the scale of the activations entering
the 12 internal normalizers on the pilot's own web and Python probe windows and on each candidate's probe sample,
using the pilot's diagnostics code unchanged. The selection statistic of a candidate is Finding A's switch-time
gap: the mean, over 3 seeds x 2 conditions x 12 sites, of |0.5 ln(E_candidate / E_web)|, with E the mean squared
norm of the normalizer input over the probe positions. The candidate with the largest statistic is selected (ties:
the fixed candidate order). Before selecting, the probe must reproduce the pilot's recorded web and Python
energies at the switch; if it cannot, it writes no selection.

  python probe_domains.py --inputs /kaggle/input --output /kaggle/working/s2probe
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_domain as PD  # noqa: E402
import s2_run as S2  # noqa: E402

REPRODUCTION_TOLERANCE = 1e-3   # relative, per site: the pilot's recorded switch energies (web, Python)
SELECTION_SCHEMA = PD.SELECTION_SCHEMA
SITES = S2.SITES


def site_energies(measurement: dict, domains) -> dict[str, dict[str, float]]:
    return {site: {domain: measurement["sites"][f"{site}.h"]["all"][domain]["mean_squared_norm"]
                   for domain in domains} for site in SITES}


def gap_statistic(energies: dict, domain: str, models) -> dict:
    """Mean |0.5 ln(E_domain / E_web)| over the given models x 12 sites, plus its per-model and per-condition parts."""
    per_model = {}
    for key in models:
        values = [abs(0.5 * math.log(energies[key][site][domain] / energies[key][site]["web"])) for site in SITES]
        per_model[f"{key[0]}/{key[1]}"] = float(np.mean(values))
    per_condition = {condition: float(np.mean([v for k, v in per_model.items() if k.endswith(f"/{condition}")]))
                     for condition in S2.CONDITIONS}
    return {"S": float(np.mean(list(per_model.values()))), "per_condition": per_condition, "per_model": per_model}


def select(statistics: dict[str, float], order=PD.CANDIDATE_ORDER) -> str:
    """Largest statistic wins; exact ties (|difference| < 1e-12) go to the earlier candidate in the fixed order."""
    best = None
    for cid in order:
        if best is None or statistics[cid] > statistics[best] + 1e-12:
            best = cid
    return best


def ranking(statistics: dict[str, float], order=PD.CANDIDATE_ORDER) -> list[str]:
    """The selection rule applied repeatedly: rank 0 is the selection, rank 1 the pre-registered fallback."""
    remaining, ranked = list(order), []
    while remaining:
        ranked.append(select(statistics, remaining))
        remaining.remove(ranked[-1])
    return ranked


def recorded_energies(pilot: S2.PilotReference, seed: int, condition: str) -> dict:
    record = pilot.events(seed, condition).get("prefix", S2.PREFIX_END, "diag", "both")
    if record is None or record.get("status") != "ok":
        raise ValueError(f"The pilot has no switch diagnostics for S{seed}-{condition}")
    return site_energies(record["measurement"], ("web", "python"))


def run(inputs: Path, output: Path, cache: Path, windows_dir: Path | None, eval_microbatch: int = 16) -> dict:
    import torch
    tick = time.time()
    output.mkdir(parents=True, exist_ok=True)
    paths = locate(inputs)
    pilot = S2.PilotReference(paths["pilot_state"])
    pilot.verify()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not S2.MINI:
        raise RuntimeError("The probe must run on the pilot's hardware/precision (CUDA FP16); select GPU T4")
    S2.log(f"Pilot switch states verified; device {device} (torch {torch.__version__})")
    if windows_dir is None:
        windows_dir = output / "probe-windows"
        PD.build_probe_windows(windows_dir, cache)
    samples = PD.load_probe_windows(windows_dir)
    window_manifest = json.loads((windows_dir / "manifest.json").read_text(encoding="utf-8"))
    if set(samples) != set(PD.CANDIDATE_ORDER):
        raise ValueError(f"Probe samples for {sorted(samples)}, expected {PD.CANDIDATE_ORDER}")
    manifest = json.loads((paths["pilot_online"] / "manifest.json").read_text(encoding="utf-8"))
    classes = np.array(json.loads((paths["pilot_online"] / "classes.json").read_text(encoding="utf-8"))["classes"])
    pilot_orders = json.loads((paths["pilot_orders"] / "manifest.json").read_text(encoding="utf-8"))
    probes = {"web": S2.TrainStream(paths["pilot_online"] / manifest["splits"]["web_train"]["file"]),
              "python": S2.TrainStream(paths["pilot_online"] / manifest["splits"]["python_train"]["file"])}
    for cid in PD.CANDIDATE_ORDER:
        full = S2.WindowArray(samples[cid])
        half = (full.windows // 2) * S2.CONTEXT
        probes[cid] = full
        probes[f"{cid}#first-half"] = S2.WindowArray(samples[cid][:half + 1])
        probes[f"{cid}#second-half"] = S2.WindowArray(samples[cid][half:])
    energies, reproduction = {}, {}
    for seed in S2.PILOT_SEEDS:
        rare = np.load(paths["pilot_orders"] / pilot_orders["seeds"][str(seed)]["rare"]["file"], allow_pickle=False)
        for condition in S2.CONDITIONS:
            model, optimizer, scaler = S2.make_state(seed, condition, device)
            pilot.load_switch(seed, condition, model, optimizer, scaler)
            measurement = S2.diagnostics(model, probes, classes, rare, eval_microbatch, device)
            energies[seed, condition] = site_energies(measurement, list(probes))
            recorded = recorded_energies(pilot, seed, condition)
            diffs = [abs(energies[seed, condition][site][d] - recorded[site][d]) / abs(recorded[site][d])
                     for site in SITES for d in ("web", "python")]
            reproduction[f"{seed}/{condition}"] = max(diffs)
            S2.log(f"S{seed}-{condition}: reproduction max rel diff {max(diffs):.2e}")
            del model, optimizer, scaler
            if device.type == "cuda":
                torch.cuda.empty_cache()
    models = [(s, c) for s in S2.PILOT_SEEDS for c in S2.CONDITIONS]
    reproduced = max(reproduction.values()) <= REPRODUCTION_TOLERANCE
    candidates = {}
    for cid in PD.CANDIDATE_ORDER:
        candidate = PD.CANDIDATES[cid]
        stats = gap_statistic(energies, cid, models)
        halves = [gap_statistic(energies, f"{cid}#{part}", models)["S"] for part in ("first-half", "second-half")]
        info = window_manifest["candidates"][cid]
        candidates[cid] = {"label": candidate.label, "repo": candidate.repo, "revision": candidate.revision,
                           **stats, "half_sample_S": halves,
                           "probe_sample": {k: info[k] for k in ("tokens", "tokens_sha256", "file_sha256")}
                           | {"documents": len(info["documents"]), "first_shard": info["documents"][0]["shard"]}}
    python = gap_statistic(energies, "python", models)
    ranked = ranking({cid: row["S"] for cid, row in candidates.items()})
    selected = ranked[0] if reproduced else None
    result = {
        "schema": SELECTION_SCHEMA, "status": "selected" if reproduced else "reproduction_failed",
        "selected": selected,
        "rule": "argmax over candidates of the mean over 3 seeds x 2 conditions x 12 internal normalizer inputs of "
                "|0.5 ln(E_candidate / E_web)| at the pilot switch states (E = mean squared norm over 256 probe "
                "windows); ties -> fixed candidate order",
        "candidate_order": list(PD.CANDIDATE_ORDER), "candidates": candidates, "ranking": ranked,
        "python_reference": python | {"note": "the pilot's Python probe windows; Finding A reported 0.097 (RMS) and "
                                              "0.123 (Taper-minus)"},
        "selected_exceeds_python": None if selected is None else candidates[selected]["S"] > python["S"],
        "reproduction": {"tolerance_relative": REPRODUCTION_TOLERANCE, "max_relative_difference": reproduction,
                         "passed": reproduced},
        "energies": {f"{s}/{c}": energies[s, c] for s, c in models},
        "pilot_switch_sha256": pilot.pins["switch_sha256"],
        "probe_windows_manifest_sha256": PD.sha256_file(windows_dir / "manifest.json"),
        "code_sha256": {name: PD.sha256_file(Path(__file__).resolve().parent / name)
                        for name in ("probe_domains.py", "prepare_domain.py", "s2_run.py")},
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device),
                        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "eval_microbatch": eval_microbatch, "seconds": time.time() - tick}
    PD.json_write(output / "selection.json", result)
    lines = [f"SELECTED: {selected} ({result['status']})" if selected else f"NO SELECTION: {result['status']}",
             f"Python reference S = {python['S']:.4f} (RMS {python['per_condition']['RMS']:.4f}, "
             f"Taper-minus {python['per_condition']['Taper-minus']:.4f})"]
    for cid in result["ranking"]:
        row = candidates[cid]
        lines.append(f"{cid:12s} S = {row['S']:.4f} (RMS {row['per_condition']['RMS']:.4f}, Taper-minus "
                     f"{row['per_condition']['Taper-minus']:.4f}; halves {row['half_sample_S'][0]:.4f} / "
                     f"{row['half_sample_S'][1]:.4f})")
    lines.append(f"Reproduction of the pilot's switch energies: max rel diff {max(reproduction.values()):.2e} "
                 f"({'passed' if reproduced else 'FAILED'})")
    (output / "selection.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return result


def locate(inputs: Path) -> dict[str, Path]:
    """The pilot dataset (online/ + orders/) and the pilot run's state (h1state/), found by content."""
    pilot_online = sorted({p.parent.resolve() for p in inputs.rglob("manifest.json") if p.parent.name == "online"
                           and json.loads(p.read_text(encoding="utf-8")).get("schema") == "pilot-arrays-v1"})
    pilot_state = sorted({p.parent.resolve() for p in inputs.rglob(S2.PILOT_STATE_MARKER)})
    if len(pilot_online) != 1 or len(pilot_state) != 1:
        raise FileNotFoundError(f"Attach the pilot dataset and the pilot run notebook: {pilot_online}, {pilot_state}")
    return {"pilot_online": pilot_online[0], "pilot_orders": pilot_online[0].parent / "orders",
            "pilot_state": pilot_state[0]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/s2probe"))
    parser.add_argument("--cache", type=Path, default=Path("/tmp/s2probe-cache"))
    parser.add_argument("--windows", type=Path, default=None, help="reuse probe samples built earlier")
    parser.add_argument("--eval-microbatch", type=int, default=16)
    args = parser.parse_args(argv)
    result = run(args.inputs, args.output, args.cache, args.windows, args.eval_microbatch)
    return 0 if result["status"] == "selected" else 8


if __name__ == "__main__":
    sys.exit(main())
