import pytest

from domain_shift_forgetting.evaluation.statistics import EvaluationSums
from domain_shift_forgetting.training.budget import BudgetProjection, must_checkpoint_and_stop
from domain_shift_forgetting.analysis.contributions import decompose


def test_loss_weighting_and_rare_overlap():
    sums = EvaluationSums()
    sums.add([1, 2, 3, 4], ["a", "a", "a", "b"], ["W", "A", "P", "X"], [True, False, True, False])
    sums.check_reconstruction()
    assert sums.total.mean == 2.5
    assert sums.code_ce() == 3
    assert sums.code_ce(alphanumeric_punctuation_only=True) == 2.5
    assert sums.rare.count == 2 and sums.total.count == 4
    assert EvaluationSums().code_ce() is None


def test_budget_includes_prior_failures_and_save_allowance():
    assert BudgetProjection(0, 20 * 3600, 0, 0).fits
    assert not BudgetProjection(2 * 3600, 20 * 3600, 0, 0).fits
    assert must_checkpoint_and_stop(total_billed_seconds=24 * 3600 - 60, save_allowance_seconds=60)
    with pytest.raises(ValueError):
        BudgetProjection(float("nan"), 0, 0, 0)


def test_four_way_statistics_reconstruct_disjoint_contributions():
    groups = []
    for losses in ([1.2, 2.4], [1.0, 2.0], [1.1, 2.1], [1.0, 2.0]):
        group = EvaluationSums()
        group.add(losses, ["a", "b"], ["W", "X"], [True, False])
        groups.append(group)
    d, documents, classes = decompose(*groups)
    assert d == pytest.approx(0.2)
    assert sum(c.weighted_contribution for c in classes) == pytest.approx(d)
    assert sum(r.count * r.d for r in documents) / 2 == pytest.approx(d)
    groups[0].documents["a"].count += 1
    with pytest.raises(ValueError):
        decompose(*groups)
