import torch
from torch.nn import functional as F

from puzzel.model.config import ModelConfig
from puzzel.model.transformer import TransformerLM, count_parameters


def _tiny_model() -> TransformerLM:
    config = ModelConfig(
        vocab_size=260,
        context_length=16,
        d_model=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    )
    return TransformerLM(config)


def test_forward_shape_initialization_and_gradients() -> None:
    torch.manual_seed(17)
    model = _tiny_model()
    token_ids = torch.tensor([[1, 2, 3, 4], [7, 8, 9, 10]], dtype=torch.long)

    logits = model(token_ids)
    targets = torch.tensor([[2, 3, 4, 5], [8, 9, 10, 11]], dtype=torch.long)
    loss = F.cross_entropy(logits.reshape(-1, 260), targets.reshape(-1))
    loss.backward()

    assert logits.shape == (2, 4, 260)
    assert torch.isfinite(loss)
    assert model.parameter_count() > 0
    assert count_parameters(model) == model.parameter_count()
    assert model.verify_random_initialization()
    assert model.token_embedding.weight.grad is not None
    assert torch.count_nonzero(model.token_embedding.weight).item() > 0
    assert model.language_model_head.weight is model.token_embedding.weight


def test_future_tokens_do_not_change_earlier_logits() -> None:
    model = _tiny_model().eval()
    first = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    changed_future = torch.tensor([[1, 2, 99, 100]], dtype=torch.long)
    mask = model.blocks[0].attention.causal_mask
    expected_mask = torch.tril(torch.ones_like(mask))

    with torch.no_grad():
        first_logits = model(first)
        changed_logits = model(changed_future)

    assert torch.equal(mask, expected_mask)
    torch.testing.assert_close(first_logits[:, :2], changed_logits[:, :2])


def test_random_initialization_depends_on_rng_seed() -> None:
    torch.manual_seed(101)
    first_model = _tiny_model()
    torch.manual_seed(202)
    second_model = _tiny_model()

    assert first_model.verify_random_initialization()
    assert second_model.verify_random_initialization()
    assert not torch.equal(
        first_model.token_embedding.weight, second_model.token_embedding.weight
    )


def test_fixed_shape_torch_export_smoke() -> None:
    model = _tiny_model().eval()
    token_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    exported = torch.export.export(model, (token_ids,))

    with torch.no_grad():
        expected = model(token_ids)
        actual = exported.module()(token_ids)

    torch.testing.assert_close(actual, expected)
