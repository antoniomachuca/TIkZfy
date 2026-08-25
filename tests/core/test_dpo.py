import pytest
import torch

from core.exceptions import TensorTopologyError
from core.ml.dpo import DirectPreferenceOptimizationLoss
from core.ml.loss import apply_word_dropout
from core.models import UNK_INDEX


def test_apply_word_dropout_substitutes_tokens() -> None:
    torch.manual_seed(42)
    decoder_input = torch.randint(low=3, high=50, size=(4, 16), dtype=torch.long)
    dropped = apply_word_dropout(decoder_input, dropout_probability=0.5, unk_index=UNK_INDEX)

    assert dropped.shape == decoder_input.shape
    assert (dropped == UNK_INDEX).any()
    assert (dropped != UNK_INDEX).any()


def test_apply_word_dropout_rejects_invalid_inputs() -> None:
    with pytest.raises(TensorTopologyError):
        apply_word_dropout(torch.ones(10, dtype=torch.long))
    with pytest.raises(TensorTopologyError):
        apply_word_dropout(torch.ones((2, 5), dtype=torch.float32))
    with pytest.raises(TensorTopologyError):
        apply_word_dropout(torch.ones((2, 5), dtype=torch.long), dropout_probability=1.5)


def test_dpo_loss_computation() -> None:
    torch.manual_seed(10)
    batch_size, seq_len, vocab_size = 2, 8, 30
    pi_chosen = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    pi_rejected = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    ref_chosen = torch.randn(batch_size, seq_len, vocab_size)
    ref_rejected = torch.randn(batch_size, seq_len, vocab_size)
    y_chosen = torch.randint(1, vocab_size, (batch_size, seq_len))
    y_rejected = torch.randint(1, vocab_size, (batch_size, seq_len))

    dpo = DirectPreferenceOptimizationLoss(beta=0.1)
    loss = dpo(pi_chosen, pi_rejected, ref_chosen, ref_rejected, y_chosen, y_rejected)

    assert loss.ndim == 0
    assert not torch.isnan(loss)
    loss.backward()
    assert pi_chosen.grad is not None
    assert pi_rejected.grad is not None


def test_dpo_loss_rejects_topology_mismatch() -> None:
    dpo = DirectPreferenceOptimizationLoss()
    pi_chosen = torch.randn(2, 8, 30)
    pi_rejected = torch.randn(3, 8, 30)
    ref_chosen = torch.randn(2, 8, 30)
    ref_rejected = torch.randn(2, 8, 30)
    y_chosen = torch.randint(1, 30, (2, 8))
    y_rejected = torch.randint(1, 30, (2, 8))

    with pytest.raises(TensorTopologyError):
        dpo(pi_chosen, pi_rejected, ref_chosen, ref_rejected, y_chosen, y_rejected)
