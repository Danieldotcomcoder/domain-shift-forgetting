"""Small deterministic checkpoint harness exercises full state without a GPU."""

import random
import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from domain_shift_forgetting.models.norms import TaperNorm
from domain_shift_forgetting.protocol import Condition, learning_rate, gate
from domain_shift_forgetting.training.checkpoints import DataCursor, restore_complete, save_complete


class ResumeHarness(torch.nn.Module):
    condition = Condition.TAPER_MINUS

    def __init__(self):
        super().__init__()
        self.norm = TaperNorm(4)
        self.register_buffer("completed_updates", torch.zeros((), dtype=torch.int64))


def advance(model, optimizer, count):
    for _ in range(count):
        update = int(model.completed_updates) + 1
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(update)
        optimizer.zero_grad(set_to_none=True)
        inputs = torch.randn(2, 3, 4) + random.random() + float(np.random.random())
        model.norm(inputs, update=update, collect=True).square().mean().backward()
        optimizer.step()
        model.norm.finish_update(update)
        model.completed_updates.fill_(update)


def test_ten_update_resume_recovers_rng_ema_adam_and_cursor(tmp_path):
    torch.manual_seed(101)
    random.seed(101)
    np.random.seed(101)
    uninterrupted = ResumeHarness()
    opt = torch.optim.AdamW(uninterrupted.parameters(), lr=0.0006, weight_decay=0)
    advance(uninterrupted, opt, 5)
    refs = {"code_commit": "synthetic-fixture", "config_sha256": "a" * 64,
            "manifest_sha256": "b" * 64, "run_id": "test", "attempt_id": "test-1"}
    cursor = DataCursor("c" * 64, "d" * 64, 5)
    path = tmp_path / "state.pt"
    digest = save_complete(path, model=uninterrupted, optimizer=opt, cursors={"web": cursor}, references=refs)
    advance(uninterrupted, opt, 5)
    resumed = ResumeHarness()
    resumed_opt = torch.optim.AdamW(resumed.parameters(), lr=0.0006, weight_decay=0)
    restored = restore_complete(path, expected_sha256=digest, model=resumed,
                                optimizer=resumed_opt, expected_references=refs)
    assert restored["web"] == cursor
    advance(resumed, resumed_opt, 5)
    for key, value in uninterrupted.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    for a, b in zip(uninterrupted.parameters(), resumed.parameters(), strict=True):
        for key, value in opt.state.get(a, {}).items():
            torch.testing.assert_close(value, resumed_opt.state[b][key], rtol=0, atol=0)
    assert resumed_opt.param_groups[0]["lr"] == opt.param_groups[0]["lr"]
    assert gate(int(resumed.completed_updates)) == gate(10)
    with pytest.raises(FileExistsError):
        save_complete(path, model=resumed, optimizer=resumed_opt, cursors={"web": cursor}, references=refs)
