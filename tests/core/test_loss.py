import pytest
import torch
import torch.nn.functional as F
from torch import nn

from core.exceptions import TensorTopologyError
from core.math.tokenization import batch_encode, build_vocabulary
from core.ml.loss import (
    CompositeMultiTaskLoss,
    CompositeMultiTaskLossV4,
    GaussianOrdinalCoordinateLoss,
    LossComponents,
    SpatialAwareHybridLoss,
    TeacherForcingCrossEntropy,
    build_adamw_optimizer,
    build_cosine_warmup_scheduler,
    build_teacher_forcing_pair,
    build_token_loss_weights,
    warmup_cosine_ratio,
)
from core.ml.model import VisionAutoregressiveModel
from core.models import PAD_INDEX, TikzTokens

SAMPLE_MARKUP: str = r"\begin{tikzpicture}\draw (0,0) -- (1,1);\end{tikzpicture}"


def _tiny_model() -> VisionAutoregressiveModel:
    torch.manual_seed(11)
    vocabulary = build_vocabulary([TikzTokens(markup=SAMPLE_MARKUP)])
    return VisionAutoregressiveModel(
        vocabulary=vocabulary,
        input_channels=3,
        model_dimension=32,
        max_length=64,
        num_layers=1,
        num_heads=4,
    )


def test_teacher_forcing_pair_shifts_targets() -> None:
    tokens = torch.tensor([[1, 5, 7, 2, 0, 0], [1, 4, 2, 0, 0, 0]])

    decoder_input, targets = build_teacher_forcing_pair(tokens)

    assert decoder_input.shape == targets.shape == (2, 5)
    assert torch.equal(decoder_input, tokens[:, :-1])
    assert torch.equal(targets, tokens[:, 1:])
    assert targets[0, 0].item() == 5


def test_teacher_forcing_pair_rejects_invalid_tensors() -> None:
    with pytest.raises(TensorTopologyError):
        build_teacher_forcing_pair(torch.ones(8, dtype=torch.long))
    with pytest.raises(TensorTopologyError):
        build_teacher_forcing_pair(torch.ones(2, 8, dtype=torch.float32))
    with pytest.raises(TensorTopologyError):
        build_teacher_forcing_pair(torch.ones(2, 1, dtype=torch.long))


def test_cross_entropy_ignores_padding_positions() -> None:
    torch.manual_seed(3)
    logits = torch.randn(2, 6, 11)
    targets = torch.tensor([[1, 5, 7, 2, 0, 0], [1, 4, 2, 0, 0, 0]])
    criterion = TeacherForcingCrossEntropy(ignore_index=PAD_INDEX)

    baseline = criterion(logits, targets)
    perturbed = logits.clone()
    perturbed[targets.eq(PAD_INDEX)] = 1e6

    assert torch.allclose(baseline, criterion(perturbed, targets))


def test_cross_entropy_matches_manual_token_average() -> None:
    torch.manual_seed(5)
    logits = torch.randn(2, 5, 7)
    targets = torch.tensor([[3, 4, 2, 0, 0], [5, 6, 2, 0, 0]])
    criterion = TeacherForcingCrossEntropy(ignore_index=PAD_INDEX)

    non_padding = targets.ne(PAD_INDEX)
    manual = F.cross_entropy(logits[non_padding], targets[non_padding])

    assert torch.allclose(criterion(logits, targets), manual)


def test_cross_entropy_approaches_zero_for_perfect_prediction() -> None:
    targets = torch.tensor([[4, 2, 0]])
    logits = torch.full((1, 3, 9), -10.0)
    logits[0, 0, 4] = 10.0
    logits[0, 1, 2] = 10.0
    logits[0, 2, 3] = 10.0

    loss = TeacherForcingCrossEntropy()(logits, targets)

    assert loss.item() < 1e-4


def test_cross_entropy_rejects_shape_mismatch() -> None:
    criterion = TeacherForcingCrossEntropy()
    with pytest.raises(TensorTopologyError):
        criterion(torch.randn(2, 5), torch.ones(2, 5, dtype=torch.long))
    with pytest.raises(TensorTopologyError):
        criterion(torch.randn(2, 5, 7), torch.ones(2, 4, dtype=torch.long))
    with pytest.raises(TensorTopologyError):
        criterion(torch.randn(2, 5, 7), torch.ones(2, 5, dtype=torch.float32))


def test_adamw_separates_weight_decay_parameter_groups() -> None:
    model = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))

    optimizer = build_adamw_optimizer(model, learning_rate=1e-3, weight_decay=0.1)

    decay_group, no_decay_group = optimizer.param_groups
    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0
    assert all(parameter.ndim >= 2 for parameter in decay_group["params"])
    assert all(parameter.ndim < 2 for parameter in no_decay_group["params"])


def test_adamw_applies_decoupled_weight_decay_without_gradients() -> None:
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = build_adamw_optimizer(model, learning_rate=0.1, weight_decay=0.5)

    loss: torch.Tensor = model.weight.sum() * 0.0
    loss.backward()  # type: ignore[no-untyped-call]
    optimizer.step()

    expected = 1.0 * (1.0 - 0.1 * 0.5)
    assert torch.allclose(model.weight.detach(), torch.full((2, 2), expected), atol=1e-7)


def test_adamw_rejects_invalid_hyperparameters() -> None:
    model = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="learning_rate"):
        build_adamw_optimizer(model, learning_rate=0.0)
    with pytest.raises(ValueError, match="weight_decay"):
        build_adamw_optimizer(model, weight_decay=-0.1)
    with pytest.raises(ValueError, match="betas"):
        build_adamw_optimizer(model, betas=(1.5, 0.999))
    with pytest.raises(ValueError, match="epsilon"):
        build_adamw_optimizer(model, epsilon=0.0)


def test_warmup_cosine_ratio_hits_warmup_peak_and_floor() -> None:
    assert warmup_cosine_ratio(0, warmup_steps=4, total_steps=14) == pytest.approx(0.25)
    assert warmup_cosine_ratio(3, warmup_steps=4, total_steps=14) == pytest.approx(1.0)
    assert warmup_cosine_ratio(14, warmup_steps=4, total_steps=14, min_lr_ratio=0.1) == (
        pytest.approx(0.1)
    )


def test_scheduler_linear_warmup_then_cosine_decay() -> None:
    optimizer = build_adamw_optimizer(nn.Linear(4, 4), learning_rate=1.0, weight_decay=0.0)
    scheduler = build_cosine_warmup_scheduler(
        optimizer, warmup_steps=4, total_steps=14, min_lr_ratio=0.1
    )

    learning_rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(14):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])

    assert learning_rates[0] == pytest.approx(0.25)
    assert learning_rates[1] == pytest.approx(0.5)
    assert learning_rates[3] == pytest.approx(1.0)
    assert learning_rates[-1] == pytest.approx(0.1)
    decay_phase = learning_rates[3:]
    assert decay_phase == sorted(decay_phase, reverse=True)


def test_scheduler_rejects_invalid_step_configuration() -> None:
    optimizer = build_adamw_optimizer(nn.Linear(2, 2))
    with pytest.raises(ValueError, match="warmup_steps"):
        build_cosine_warmup_scheduler(optimizer, warmup_steps=0, total_steps=10)
    with pytest.raises(ValueError, match="total_steps"):
        build_cosine_warmup_scheduler(optimizer, warmup_steps=10, total_steps=10)
    with pytest.raises(ValueError, match="min_lr_ratio"):
        build_cosine_warmup_scheduler(optimizer, warmup_steps=2, total_steps=10, min_lr_ratio=1.5)


def test_training_steps_reduce_teacher_forced_loss() -> None:
    model = _tiny_model()
    corpus = [TikzTokens(markup=SAMPLE_MARKUP)] * 2
    tokens = batch_encode(corpus, model.vocabulary, max_length=64)
    decoder_input, targets = build_teacher_forcing_pair(tokens)
    images = torch.randn(2, 3, 32, 32)
    criterion = TeacherForcingCrossEntropy()
    optimizer = build_adamw_optimizer(model, learning_rate=1e-2)
    scheduler = build_cosine_warmup_scheduler(optimizer, warmup_steps=2, total_steps=6)

    initial_loss = criterion(model(images, decoder_input), targets).item()
    for _ in range(5):
        optimizer.zero_grad()
        loss = criterion(model(images, decoder_input), targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
    final_loss = criterion(model(images, decoder_input), targets).item()

    assert final_loss < initial_loss


def test_build_token_loss_weights_and_weighted_loss() -> None:
    vocab = build_vocabulary([TikzTokens(markup=SAMPLE_MARKUP)])
    weights = build_token_loss_weights(
        vocab, coordinate_weight=5.0, geometric_weight=3.0, boilerplate_weight=0.5
    )

    assert weights.shape == (len(vocab.token_to_index),)
    assert (weights > 0).all()

    criterion = SpatialAwareHybridLoss(vocabulary=vocab, spatial_weight=1.0)
    logits = torch.randn(2, 6, len(vocab.token_to_index))
    targets = torch.randint(0, len(vocab.token_to_index), (2, 6))

    loss = criterion(logits, targets)
    assert loss.ndim == 0
    assert not torch.isnan(loss)


def test_gaussian_ordinal_loss_shapes_and_invariants() -> None:
    vocab = build_vocabulary([TikzTokens(markup=SAMPLE_MARKUP)])
    criterion = GaussianOrdinalCoordinateLoss(vocabulary=vocab, sigma=0.2)

    logits = torch.randn(2, 6, len(vocab.token_to_index), requires_grad=True)
    targets = torch.randint(0, len(vocab.token_to_index), (2, 6))

    loss = criterion(logits, targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert float(loss.item()) >= 0.0

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()

    with pytest.raises(ValueError, match="sigma"):
        GaussianOrdinalCoordinateLoss(vocabulary=vocab, sigma=0.0)


def test_composite_multitask_loss_all_components() -> None:
    vocab = build_vocabulary([TikzTokens(markup=SAMPLE_MARKUP)])
    criterion = CompositeMultiTaskLoss(
        vocabulary=vocab,
        lambda_coord=1.0,
        lambda_family=1.5,
        lambda_spatial=2.0,
        label_smoothing=0.05,
        sigma=0.2,
    )

    batch_size: int = 3
    seq_len: int = 8
    vocab_size: int = len(vocab.token_to_index)
    num_families: int = 8

    token_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    target_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    family_logits = torch.randn(batch_size, num_families, requires_grad=True)
    family_targets = torch.tensor([0, 2, 6], dtype=torch.long)

    components = criterion(
        token_logits=token_logits,
        target_tokens=target_tokens,
        family_logits=family_logits,
        family_targets=family_targets,
        return_components=True,
    )
    assert isinstance(components, LossComponents)
    assert torch.isfinite(components.total_loss)
    assert torch.isfinite(components.syntax_loss)
    assert torch.isfinite(components.gaussian_ord_loss)
    assert torch.isfinite(components.huber_loss)
    assert torch.isfinite(components.family_loss)

    expected_total = (
        components.syntax_loss
        + 1.0 * components.gaussian_ord_loss
        + 2.0 * components.huber_loss
        + 1.5 * components.family_loss
    )
    assert torch.allclose(components.total_loss, expected_total, atol=1e-5)

    components.total_loss.backward()  # type: ignore[no-untyped-call]
    assert token_logits.grad is not None
    assert torch.isfinite(token_logits.grad).all()
    assert family_logits.grad is not None
    assert torch.isfinite(family_logits.grad).all()


def test_composite_multitask_loss_fallback_when_unsupervised_family() -> None:
    vocab = build_vocabulary([TikzTokens(markup=SAMPLE_MARKUP)])
    criterion = CompositeMultiTaskLossV4(vocabulary=vocab)

    token_logits = torch.randn(2, 6, len(vocab.token_to_index))
    target_tokens = torch.randint(0, len(vocab.token_to_index), (2, 6))

    components = criterion(token_logits, target_tokens, return_components=True)
    assert isinstance(components, LossComponents)
    assert components.family_loss.item() == 0.0
    assert torch.isfinite(components.total_loss)


def test_ordinal_loss_exit_criterion_euclidean_ablation() -> None:
    """Exit criterion validation: ordinal loss reduces vertex Euclidean error by >60% vs pure CE."""
    coords = torch.linspace(-5.0, 5.0, 101)
    vocab_size = 101
    sigma = 0.2

    diff = coords.unsqueeze(1) - coords.unsqueeze(0)
    weights = torch.exp(-0.5 * (diff / sigma) ** 2)
    ordinal_matrix = weights / weights.sum(dim=1, keepdim=True)

    torch.manual_seed(42)
    target_idx = torch.tensor([60, 25, 75, 40])
    gt_coords = coords[target_idx]

    logits_ce = nn.Parameter(torch.zeros(4, vocab_size))
    logits_ord = nn.Parameter(torch.zeros(4, vocab_size))

    opt_ce = torch.optim.Adam([logits_ce], lr=0.1)
    opt_ord = torch.optim.Adam([logits_ord], lr=0.1)

    for _ in range(25):
        opt_ce.zero_grad()
        loss_ce = F.cross_entropy(logits_ce, target_idx)
        loss_ce.backward()  # type: ignore[no-untyped-call]
        opt_ce.step()

        opt_ord.zero_grad()
        log_probs = F.log_softmax(logits_ord, dim=-1)
        loss_gaussian = -(ordinal_matrix[target_idx] * log_probs).sum(dim=-1).mean()
        probs = F.softmax(logits_ord, dim=-1)
        pred_coords = (probs * coords).sum(dim=-1)
        loss_huber = F.smooth_l1_loss(pred_coords, gt_coords, beta=0.1)
        loss_total = loss_gaussian + 2.0 * loss_huber
        loss_total.backward()  # type: ignore[no-untyped-call]
        opt_ord.step()

    pred_ce = (F.softmax(logits_ce, dim=-1) * coords).sum(dim=-1)
    pred_ord = (F.softmax(logits_ord, dim=-1) * coords).sum(dim=-1)

    err_ce = (pred_ce - gt_coords).abs().mean().item()
    err_ord = (pred_ord - gt_coords).abs().mean().item()

    error_reduction = (err_ce - err_ord) / err_ce
    assert error_reduction > 0.60
