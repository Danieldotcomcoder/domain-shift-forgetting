import pytest

torch = pytest.importorskip("torch")

from domain_shift_forgetting.models.norms import RMSNorm, TaperNorm


def test_rms_endpoint_and_inactive_optimizer_state():
    rms, taper = RMSNorm(4), TaperNorm(4)
    h = torch.tensor([[[1., 2., 3., 4.], [4., 3., 2., 1.]]])
    optimizer = torch.optim.AdamW(taper.parameters(), lr=0.001, weight_decay=0)
    actual = taper(h, update=1, collect=True)
    torch.testing.assert_close(actual, rms(h), rtol=1e-5, atol=1e-6)
    actual.square().mean().backward()
    optimizer.step()
    taper.finish_update(1)
    assert taper.gamma_tilde.grad is None
    assert taper.gamma_tilde not in optimizer.state
    assert int(taper.ema_updates) == 1


def test_same_sample_ema_freeze_and_first_active_gain_step():
    taper = TaperNorm(3)
    optimizer = torch.optim.AdamW(taper.parameters(), lr=0.001, weight_decay=0)
    numerator = denominator = 0.0
    for update in range(1, 764):
        optimizer.zero_grad(set_to_none=True)
        h = torch.tensor([[[1., 2., 3.]], [[2., 3., 4.]]]) * (1 + update / 1000)
        r = (h.square().mean(-1) + 1e-6).sqrt()
        energy = (h * taper.gamma.detach()).square().sum(-1)
        numerator = 0.99 * numerator + 0.01 * float((energy / r).mean())
        denominator = 0.99 * denominator + 0.01 * float(energy.mean())
        # Two microbatches must yield just one update-level EMA step.
        for micro in h.split(1):
            taper(micro, update=update, collect=True).square().mean().div(2).backward()
        optimizer.step()
        taper.finish_update(update)
    correction = 1 - 0.99 ** 763
    expected = (numerator / correction) / (denominator / correction + 1e-12)
    assert float(taper.c) == pytest.approx(expected, rel=1e-5)
    torch.testing.assert_close(taper.gamma, taper.gamma_tilde)
    assert taper.gamma_tilde not in optimizer.state
    optimizer.zero_grad(set_to_none=True)
    taper(h, update=764).square().mean().backward()
    optimizer.step()
    assert int(optimizer.state[taper.gamma_tilde]["step"]) == 1
    assert int(optimizer.state[taper.gamma]["step"]) == 764


def test_zero_gate_linear_equivalence_and_inactive_gamma():
    taper = TaperNorm(4)
    taper.calibrated.fill_(True)
    taper.c.fill_(0.7)
    h = torch.tensor([[[1., 2., 3., 4.]]], requires_grad=True)
    y = taper(h, update=6104)
    torch.testing.assert_close(y, h * taper.c * taper.gamma_tilde, rtol=1e-5, atol=1e-6)
    y.sum().backward()
    assert taper.gamma.grad is None
    assert taper.gamma_tilde.grad is not None
