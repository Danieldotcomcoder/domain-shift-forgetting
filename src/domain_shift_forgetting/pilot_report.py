"""Fixed-endpoint reports; incomplete runs cannot become negative findings."""
from dataclasses import asdict
from pathlib import Path

from .analysis.contrasts import BranchCE, contrast, seed_uncertainty, prefix_slope
from .analysis.contributions import decompose
from .analysis.decision import DecisionThresholds, SeedEvidence, classify
from .analysis.matching import AdaptationPoint, matched_forgetting
from .analysis.tails import summarize_documents
from .evaluation.statistics import EvaluationSums, LossSum
from .pilot_control import read_json, write_json, digest_json
from .protocol import SEEDS, QUICK_CONTINUATION, FULL_CONTINUATION, FULL_PREFIX, PREFIX_END, CONTINUATION_UPDATES, DIAGNOSTIC_CONTINUATION


def condition_root(root, condition):
    return root / "workers" / condition if (root / "workers").is_dir() else root


def event_path(root, seed, condition, stage, step, role, domain):
    return condition_root(root, condition) / "events" / f"S{seed}-{condition}" / f"{stage}-{step}-{role}-{domain}.json"


def sufficient_stats(row: dict) -> EvaluationSums:
    s = row["measurement"]["statistics"]
    return EvaluationSums(total=LossSum(**s["total"]), rare=LossSum(**s["rare"]),
                          documents={k: LossSum(**v) for k, v in s["documents"].items()},
                          classes={k: LossSum(**v) for k, v in s["classes"].items()})


def report(root: Path) -> dict:
    freeze = read_json(root / "freeze.json")
    freeze_hash = digest_json(freeze)
    missing, invalid, seeds, details, observations = [], [], [], [], []

    def get(seed, condition, stage, step, role, domain):
        path = event_path(root, seed, condition, stage, step, role, domain)
        if not path.exists():
            missing.append(str(path.relative_to(root)))
            return None
        row = read_json(path)
        if (row.get("freeze_sha256") != freeze_hash or row.get("seed") != seed or row.get("condition") != condition
                or row.get("stage") != stage or row.get("step") != step
                or row["measurement"]["role"] != role or row["measurement"]["domain"] != domain
                or row["measurement"]["array_sha256"] != freeze["arrays"]["splits"][f"{domain}_dev"]["sha256"]):
            raise ValueError(f"Evaluation provenance mismatch: {path}")
        expected_update = step if stage == "prefix" else PREFIX_END + step
        if row.get("global_update") != expected_update:
            raise ValueError("Evaluation update clock mismatch")
        if stage != "prefix":
            switch = condition_root(root, condition) / "runs" / f"S{seed}-{condition}" / "switch.pt.json"
            if not switch.exists() or row.get("parent_sha256") != read_json(switch)["sha256"]:
                raise ValueError("Branch evaluation does not descend from the frozen prefix")
        elif row.get("parent_sha256") is not None:
            raise ValueError("Prefix unexpectedly has a branch parent")
        expected = 2_097_152 if role == "full" else 262_144
        stats = sufficient_stats(row)
        stats.check_reconstruction()
        if stats.total.count != expected:
            raise ValueError("Wrong evaluation label population")
        if abs(stats.total.mean - row["measurement"]["ce"]) > 1e-9:
            raise ValueError("CE does not match sufficient statistics")
        for field, expected_ce in (("non_w_ce", stats.code_ce()), ("ap_ce", stats.code_ce(alphanumeric_punctuation_only=True))):
            if expected_ce is None or abs(expected_ce - row["measurement"][field]) > 1e-9:
                raise ValueError("Code CE does not match class statistics")
        return row

    def value(row, metric="ce"):
        return row["measurement"][metric]

    for seed in SEEDS:
        cache = {}
        for condition in ("RMS", "Taper-minus"):
            completion = condition_root(root, condition) / "runs" / f"S{seed}-{condition}" / "completion.json"
            if not completion.exists():
                missing.append(str(completion.relative_to(root)))
            elif read_json(completion).get("freeze_sha256") != freeze_hash:
                raise ValueError("Completion freeze mismatch")
            for stage, points in (("prefix", (PREFIX_END,)), ("web", DIAGNOSTIC_CONTINUATION), ("python", DIAGNOSTIC_CONTINUATION)):
                for step in points:
                    path = condition_root(root, condition) / "diagnostics" / f"S{seed}-{condition}" / f"{stage}-{step}.json"
                    if not path.exists():
                        missing.append(str(path.relative_to(root)))
                    elif read_json(path).get("freeze_sha256") != freeze_hash:
                        raise ValueError("Diagnostic freeze mismatch")
            for stage, steps in (("prefix", FULL_PREFIX), ("web", FULL_CONTINUATION), ("python", FULL_CONTINUATION)):
                for step in steps:
                    for domain in ("web", "python"):
                        cache[condition, stage, step, "full", domain] = get(seed, condition, stage, step, "full", domain)
            for stage, steps in (("prefix", (PREFIX_END,)), ("web", QUICK_CONTINUATION), ("python", QUICK_CONTINUATION)):
                for step in steps:
                    for domain in ("web", "python"):
                        cache[condition, stage, step, "quick", domain] = get(seed, condition, stage, step, "quick", domain)
        for condition in ("RMS", "Taper-minus"):
            for stage in ("prefix", "web", "python"):
                points = [(key[2], row) for key, row in cache.items()
                          if key[0] == condition and key[1] == stage and key[3:] == ("full", "web") and row is not None]
                if points:
                    step, row = max(points, key=lambda pair: pair[0])
                    code_row = cache[condition, stage, step, "full", "python"]
                    observations.append({"seed": seed, "condition": condition, "stage": stage, "step": step,
                        "web_full_ce": value(row), "python_non_w_full_ce": value(code_row, "non_w_ce") if code_row else None,
                        "note": "Progress observation; not an early go/no-go decision"})
        if any(row is None for row in cache.values()):
            continue

        def c(condition, stage, step, role="full", domain="web"):
            return cache[condition, stage, step, role, domain]

        def effect(step, role="full"):
            # Prefixes cancel from D but are required for actual forgetting/Q.
            return contrast(BranchCE(
                value(c("RMS", "prefix", PREFIX_END, role)), value(c("Taper-minus", "prefix", PREFIX_END, role)),
                value(c("RMS", "web", step, role)), value(c("RMS", "python", step, role)),
                value(c("Taper-minus", "web", step, role)), value(c("Taper-minus", "python", step, role))))

        endpoint = effect(CONTINUATION_UPDATES)
        points = {condition: [AdaptationPoint(step,
            value(c(condition, "python", step, "quick", "python"), "non_w_ce"),
            value(c(condition, "python", step, "quick", "web"))) for step in QUICK_CONTINUATION]
            for condition in ("RMS", "Taper-minus")}
        matches = {u: matched_forgetting(points["RMS"], points["Taper-minus"], u) for u in (1525, 3050, 6104)}
        prefix_rms = value(c("RMS", "prefix", PREFIX_END))
        prefix_taper = value(c("Taper-minus", "prefix", PREFIX_END))
        if prefix_rms <= 0:
            raise ValueError("Invalid prefix reference denominator")
        adaptation = {condition: value(c(condition, "prefix", PREFIX_END, domain="python"), "non_w_ce") -
            value(c(condition, "python", CONTINUATION_UPDATES, domain="python"), "non_w_ce")
            for condition in ("RMS", "Taper-minus")}
        evidence = SeedEvidence(seed, endpoint.d, effect(3050).d, endpoint.q,
            abs(prefix_taper - prefix_rms) / prefix_rms, adaptation["RMS"], adaptation["Taper-minus"],
            {u: effect(u, "quick").d for u in QUICK_CONTINUATION}, {u: pair[1] for u, pair in matches.items()})
        seeds.append(evidence)
        d, documents, classes = decompose(*(sufficient_stats(c(condition, branch, 6104))
            for condition, branch in (("Taper-minus", "python"), ("Taper-minus", "web"), ("RMS", "python"), ("RMS", "web"))))
        if abs(d - endpoint.d) > 1e-6:
            raise ValueError("Document decomposition disagrees with primary contrast")
        slopes = {condition: prefix_slope([(u, value(c(condition, "prefix", u))) for u in FULL_PREFIX])
                  for condition in ("RMS", "Taper-minus")}
        details.append({"seed": seed, "endpoint": asdict(endpoint), "evidence": asdict(evidence),
            "full_dev_persistence": {u: asdict(effect(u)) for u in (1525, 3050, 6104)},
            "matched": {u: {"match": asdict(pair[0]) if pair[0] else None, "differential_forgetting": pair[1]}
                        for u, pair in matches.items()}, "prefix_slopes_nats_per_million_tokens": slopes,
            "prefix_slope_difference": slopes["Taper-minus"] - slopes["RMS"],
            "tail": asdict(summarize_documents(documents, d)), "class_contributions": [asdict(x) for x in classes]})
    if (root / "failure.json").exists():
        invalid.append(read_json(root / "failure.json"))
    for condition in ("RMS", "Taper-minus"):
        failure = condition_root(root, condition) / "failure.json"
        if failure != root / "failure.json" and failure.exists():
            invalid.append(read_json(failure))
    decision = classify(seeds, primary_complete=not missing and len(seeds) == 3,
        correctness_and_data_passed=freeze.get("admitted") is True and not invalid,
        nonfinite_primary=bool(invalid), thresholds=DecisionThresholds(**freeze["policy"]["decision_thresholds"]))
    result = {"protocol": freeze["policy"]["protocol"], "freeze_sha256": freeze_hash,
        "decision": asdict(decision), "completed_seed_groups": len(seeds), "missing_event_count": len(missing),
        "missing_events": missing, "failures": invalid, "seeds": details, "latest_observations": observations,
        "uncertainty": asdict(seed_uncertainty([s.d for s in seeds])) if len(seeds) == 3 else None,
        "thresholds": freeze["policy"]["decision_thresholds"],
        "interpretation": "Fixed-sample development-data screening, not confirmation or equivalence. "
                          "Twenty elapsed hours alone never imply a small effect. No outcome-driven extension."}
    write_json(root / "decision-report.json", result)
    lines = [f"Decision: {decision.category}", f"Completed paired seeds: {len(seeds)}/3", f"Missing events: {len(missing)}"]
    for seed in seeds:
        lines.append(f"Seed {seed.seed}: D={seed.d:.6f}; D3050={seed.d3050:.6f}; Q={seed.q:.6f}; prefix gap={seed.absolute_relative_prefix_gap:.2%}; code improvements RMS={seed.rms_non_w_improvement:.6f}, Taper={seed.taper_non_w_improvement:.6f}")
    lines += list(decision.reasons)
    (root / "decision-report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
