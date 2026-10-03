import json
import random

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from domain_shift_forgetting.kaggle import run
from domain_shift_forgetting.local_training import save, restore


def test_atomic_generation_rng_integrity_and_previous(tmp_path, monkeypatch):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    generator = torch.Generator().manual_seed(3)
    path = tmp_path / "latest.pt"
    identity = {"fixture": True}
    save(path, model, optimizer, scaler, generator, {"step": 0}, identity)
    expected = (random.random(), np.random.random(), torch.rand(1))
    restore(path, model, optimizer, scaler, generator, identity)
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    assert torch.equal(torch.rand(1), expected[2])
    original = json.loads(path.with_suffix(".pt.json").read_text())
    save(path, model, optimizer, scaler, generator, {"step": 1}, identity)
    current = json.loads(path.with_suffix(".pt.json").read_text())
    assert current["previous"]["file"] == original["file"]
    assert (tmp_path / original["file"]).exists()
    # Failure before publishing leaves the previous committed checkpoint usable.
    def fail(*args, **kwargs):
        raise OSError("simulated interrupted write")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError):
        save(path, model, optimizer, scaler, generator, {"step": 2}, identity)
    assert restore(path, model, optimizer, scaler, generator, identity)["step"] == 1
    with (tmp_path / current["file"]).open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        restore(path, model, optimizer, scaler, generator, identity)


@pytest.mark.parametrize("case", ["rms", "taper-zero"])
def test_cpu_benchmark_exercises_replay(tmp_path, case):
    result = run(tmp_path / case, case=case, precision="fp32", updates=1,
                 batch_size=1, context=8, effective_sequences=1, device="cpu")
    assert result["resume_replay_passed"]
    assert result["final_precision_ce_absolute_difference"] == 0
    assert result["tokens_per_update"] == 8


def test_invalid_batch_rejected_before_output(tmp_path):
    with pytest.raises(ValueError):
        run(tmp_path / "invalid", batch_size=3)
    assert not (tmp_path / "invalid").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA FP16")
@pytest.mark.parametrize("case", ["rms", "taper-zero"])
def test_cuda_benchmark_replay(tmp_path, case):
    result = run(tmp_path / case, case=case, updates=1,
                 batch_size=1, context=8, effective_sequences=1)
    assert result["resume_replay_passed"]
