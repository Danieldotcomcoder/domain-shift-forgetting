import pytest

torch = pytest.importorskip("torch")

from domain_shift_forgetting.models.norms import RMSNorm, TaperNorm
from domain_shift_forgetting.models.transformer import Transformer, copy_canonical_initialization
from domain_shift_forgetting.protocol import Condition
from domain_shift_forgetting.training.step import make_optimizer


def test_canonical_copy_tied_output_and_causal_rms_endpoint():
    torch.manual_seed(101)
    rms = Transformer(Condition.RMS)
    taper = Transformer(Condition.TAPER_MINUS)
    copy_canonical_initialization(rms, taper)
    for name, parameter in rms.named_parameters():
        torch.testing.assert_close(parameter, dict(taper.named_parameters())[name], rtol=0, atol=0)
    assert isinstance(taper.final_norm, RMSNorm)
    assert sum(isinstance(m, TaperNorm) for m in taper.modules()) == 12
    assert sum(p.numel() for p in rms.parameters()) == 17718784
    assert sum(p.numel() for p in taper.parameters()) == 17721856
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        rms_logits, _ = rms(tokens)
        taper_logits, _ = taper(tokens)
        changed_logits, _ = rms(torch.tensor([[1, 2, 3, 99]]))
    torch.testing.assert_close(rms_logits, taper_logits, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(rms_logits[:, :3], changed_logits[:, :3], rtol=1e-5, atol=1e-6)
    assert rms_logits.shape == (1, 4, 50257)


def test_zero_gain_decay_and_auxiliary_evaluation_separation():
    model = Transformer(Condition.TAPER_PLUS)
    optimizer = make_optimizer(model)
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            assert group["weight_decay"] == (0.1 if parameter.ndim >= 2 else 0.0)
    # Fixture representing the already calibrated switch into update 764.
    for module in model.modules():
        if isinstance(module, TaperNorm):
            module.calibrated.fill_(True)
    model.target_frozen.fill_(True)
    model.s_target.fill_(0.5)
    tokens = torch.tensor([[1, 2]])
    with torch.no_grad():
        _, warm_aux = model(tokens, update=763, auxiliary=True)
        _, active_aux = model(tokens, update=764, auxiliary=True)
        _, eval_aux = model(tokens, update=764, auxiliary=False)
    assert warm_aux is None and eval_aux is None
    assert active_aux is not None and float(active_aux) >= 0
