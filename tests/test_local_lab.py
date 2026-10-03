import csv
import json

import pytest

torch = pytest.importorskip("torch")

from domain_shift_forgetting.local_lab import (
    TinyLM, evaluate, load_data, parser, restore_checkpoint, run, save_checkpoint, update,
)
from domain_shift_forgetting.training.step import make_optimizer


def test_local_checkpoint_replays_next_update_exactly(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    args = parser().parse_args(["--context", "16", "--batch-size", "1", "--accumulation", "2"])
    data, _ = load_data(None, 16)
    model = TinyLM()
    opt = make_optimizer(model)
    rng = torch.Generator().manual_seed(19)
    device = torch.device("cpu")
    update(model, opt, data["web_train"], args, rng, device)
    path = tmp_path / "switch.pt"
    save_checkpoint(path, model, opt, rng, 1)
    expected = update(model, opt, data["web_train"], args, rng, device)
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    assert restore_checkpoint(path, model, opt, rng) == 1
    actual = update(model, opt, data["web_train"], args, rng, device)
    assert actual == expected
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)


def test_data_is_disjoint_and_rejects_duplicate_documents(tmp_path):
    _, provenance = load_data(None, 16)
    hashes = [h for p in provenance.values() for h in p["document_sha256"]]
    assert len(set(hashes)) == len(hashes)
    for domain in ("web", "python"):
        for split in ("train", "dev"):
            folder = tmp_path / domain / split
            folder.mkdir(parents=True)
            (folder / "sample.txt").write_text("duplicate document " * 10)
    with pytest.raises(ValueError, match="Duplicate"):
        load_data(tmp_path, 16)


def test_tiny_is_causal_and_evaluation_is_batch_invariant():
    torch.set_num_threads(2)
    model = TinyLM().eval()
    with torch.no_grad():
        a, _ = model(torch.tensor([[1, 2, 3, 4]]))
        b, _ = model(torch.tensor([[1, 2, 3, 9]]))
    torch.testing.assert_close(a[:, :3], b[:, :3])
    data = torch.randint(0, 257, (7, 17))
    assert evaluate(model, data, 1, "cpu") == pytest.approx(evaluate(model, data, 3, "cpu"), abs=1e-6)


def test_accumulation_matches_single_effective_batch():
    torch.set_num_threads(2)
    torch.manual_seed(23)
    first, second = TinyLM(), TinyLM()
    second.load_state_dict(first.state_dict())
    first_opt, second_opt = make_optimizer(first), make_optimizer(second)
    data, _ = load_data(None, 16)
    small = parser().parse_args(["--batch-size", "1", "--accumulation", "2"])
    large = parser().parse_args(["--batch-size", "2", "--accumulation", "1"])
    a = update(first, first_opt, data["web_train"], small,
               torch.Generator().manual_seed(31), torch.device("cpu"))
    b = update(second, second_opt, data["web_train"], large,
               torch.Generator().manual_seed(31), torch.device("cpu"))
    assert a == pytest.approx(b, rel=1e-5, abs=1e-6)
    for left, right in zip(first.parameters(), second.parameters()):
        # Test gradient accumulation itself: Adam can amplify harmless rounding
        # differences in near-zero gradients on its first parameter update.
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-4, atol=1e-7)


def test_cpu_smoke_branches_share_switch_and_write_metrics(tmp_path):
    output = tmp_path / "run"
    args = parser().parse_args(["--device", "cpu", "--smoke", "--context", "16",
                               "--batch-size", "1", "--accumulation", "1", "--output", str(output)])
    run(args)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "completed"
    rows = list(csv.DictReader((output / "metrics.csv").open()))
    prefix = next(r for r in rows if r["stage"] == "prefix" and r["step"] == "2")
    for stage in ("web", "python"):
        start = next(r for r in rows if r["stage"] == stage and r["step"] == "0")
        assert start["web_dev_ce"] == prefix["web_dev_ce"]
        assert start["python_dev_ce"] == prefix["python_dev_ce"]
    with pytest.raises(FileExistsError):
        run(args)


def test_invalid_settings_do_not_create_output(tmp_path):
    args = parser().parse_args(["--batch-size", "0", "--output", str(tmp_path / "bad")])
    with pytest.raises(ValueError, match="positive"):
        run(args)
    assert not args.output.exists()
