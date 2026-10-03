import pytest
import json
from pathlib import Path

from domain_shift_forgetting.protocol import (
    FULL_CONTINUATION, QUICK_CONTINUATION, continuation_events, gate, learning_rate, planned_runs,
)


def test_schedule_boundaries_and_lineage():
    assert learning_rate(305) == pytest.approx(0.0006)
    assert learning_rate(15260) == pytest.approx(0.00006)
    assert gate(763) == 1
    assert 0 < gate(764) < 1
    assert gate(6104) == 0
    assert {0, 1525, 3050, 6104} <= set(FULL_CONTINUATION) & set(QUICK_CONTINUATION)
    for event in continuation_events():
        assert event.global_update == 9156 + event.continuation_update
    runs = planned_runs(False)
    assert len(runs) == 18
    assert sum(r.planned_tokens for r in runs) == 2100166656
    assert sum(r.planned_tokens for r in planned_runs(True)) == 3150249984
    assert runs[1].parent_run == runs[2].parent_run == runs[0].run_id
    assert [r.seed for r in runs[::6]] == [101, 102, 103]


def test_invalid_clock_rejected():
    for update in (0, -1, 15261):
        with pytest.raises(ValueError):
            learning_rate(update)
        with pytest.raises(ValueError):
            gate(update)


def test_draft_configuration_and_code_schedule_agree():
    config = json.loads((Path(__file__).resolve().parents[1] / "configs" / "stage1.v3.json").read_text(encoding="utf-8-sig"))
    assert config["execution_enabled"] is False
    for key, actual in (("continuation_full_union", FULL_CONTINUATION),
                        ("continuation_quick_union", QUICK_CONTINUATION)):
        spec = config["evaluation"][key]
        expected = set(spec["explicit"])
        for start, end, step in spec["inclusive_ranges"]:
            expected.update(range(start, end + 1, step))
        assert tuple(sorted(expected)) == actual
    assert config["timeline"]["primary_tokens"] == sum(r.planned_tokens for r in planned_runs(False))
