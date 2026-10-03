from dataclasses import replace

import pytest

from domain_shift_forgetting.analysis.contrasts import BranchCE, contrast, seed_uncertainty
from domain_shift_forgetting.analysis.decision import SeedEvidence, classify
from domain_shift_forgetting.analysis.matching import AdaptationPoint, first_crossing
from domain_shift_forgetting.analysis.tails import DocumentEffect, summarize_documents
from domain_shift_forgetting.protocol import QUICK_CONTINUATION


def test_exact_contrast_and_forgetting_decomposition():
    assert contrast(BranchCE(2, 2, 2, 2, 2, 2)).d == 0
    result = contrast(BranchCE(2.0, 2.01, 1.9, 2.1, 1.92, 2.2))
    assert result.d == pytest.approx(0.08)
    assert result.g_web == pytest.approx(-0.01)
    assert result.q == pytest.approx(0.09)
    assert result.q == pytest.approx(result.f_taper_python - result.f_rms_python)
    assert seed_uncertainty([0.03, 0.03, 0.03]).sample_sd == 0


def test_matching_first_crossing_exact_flat_and_unreachable():
    points = [AdaptationPoint(0, 3, 2), AdaptationPoint(50, 2, 2.1),
              AdaptationPoint(100, 3, 2.2), AdaptationPoint(150, 2, 2.3)]
    match = first_crossing(points, 2.5)
    assert match.update == 25 and match.web_ce == pytest.approx(2.05)
    assert first_crossing(points, 2).update == 50
    assert first_crossing(points, 3.5) is None
    assert first_crossing(points, 1) is None
    assert first_crossing([AdaptationPoint(0, 2, 2), AdaptationPoint(50, 2, 2)], 2).update == 0
    assert first_crossing([AdaptationPoint(0, 3, 2), AdaptationPoint(101, 2, 2.1)], 2.5) is None


def test_signed_document_contributions_and_weighted_reconstruction():
    rows = [DocumentEffect("short", 1, 0.2), DocumentEffect("long", 9, -0.01)]
    result = summarize_documents(rows, 0.011)
    assert result.d == pytest.approx(0.011)
    assert result.top_fraction_of_net > 1  # Never clamp cancellation ratios.
    assert result.top_fraction_of_positive_mass == 1


def evidence(d=0.03):
    return [SeedEvidence(s, d, 0.015, 0.025, 0.02, 0.05, 0.05,
                         {u: 0.0 for u in QUICK_CONTINUATION},
                         {1525: 0.01, 3050: 0.01, 6104: 0.01}) for s in (101, 102, 103)]


def decision(rows):
    return classify(rows, primary_complete=True, correctness_and_data_passed=True)


@pytest.mark.parametrize("d,category", [(-0.03, "OPPOSITE DIRECTION"),
    (0.0149, "STOP — SMALL OBSERVED EFFECT"), (0.015, "INCONCLUSIVE"),
    (0.03, "PROCEED TO DESIGN THE NEXT STUDY")])
def test_ordered_threshold_boundaries(d, category):
    assert decision(evidence(d)).category == category


def test_missingness_guardrails_transient_and_strict_q():
    assert decision(evidence()[:2]).category == "INVALID OR INCOMPLETE"
    rows = evidence(-0.03)
    rows[0] = replace(rows[0], absolute_relative_prefix_gap=0.02001)
    assert decision(rows).category == "COMPARABILITY / ADAPTATION LIMITED"
    rows = [replace(s, q=0.015) for s in evidence()]
    assert decision(rows).category == "INCONCLUSIVE"
    rows = [replace(s, matched_differences={u: None for u in (1525, 3050, 6104)}) for s in evidence()]
    assert decision(rows).category == "INCONCLUSIVE"
    rows = [replace(s, quick_d={**s.quick_d, 50: 0.05}) for s in evidence(0.01)]
    assert decision(rows).category == "TRANSIENT ONLY"
