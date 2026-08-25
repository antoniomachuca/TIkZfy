"""Direct Preference Optimization (DPO) objective for visual geometric alignment.

Optimizes the policy parameters directly on paired rollouts (chosen vs. rejected)
derived from compilation-based SSIM and Hungarian Graph Edit Distance rewards.

References:
    Rafailov et al., Direct Preference Optimization: Your Language Model Is
        Secretly a Reward Model (NeurIPS 2023).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from core.exceptions import TensorTopologyError
from core.models import PAD_INDEX


class DirectPreferenceOptimizationLoss(nn.Module):
    """DPO objective function comparing model log-probabilities against a frozen reference model.

    Computes:
        L_DPO = -E [ log sigma ( beta * ( log(pi/ref)_chosen - log(pi/ref)_rejected ) ) ]
    """

    def __init__(self, beta: float = 0.1, ignore_index: int = PAD_INDEX) -> None:
        super().__init__()
        if beta <= 0.0:
            raise ValueError(f"beta must be strictly positive. Got {beta}.")
        self.beta: float = beta
        self.ignore_index: int = ignore_index

    def _compute_sequence_log_probs(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute the sum log-probabilities of non-padding tokens in each sequence."""
        if logits.ndim != 3:
            raise TensorTopologyError("Logits must be rank-3 (B, L, V).")
        if targets.ndim != 2:
            raise TensorTopologyError("Targets must be rank-2 (B, L).")

        log_probs: torch.Tensor = F.log_softmax(logits, dim=-1)  # Shape: (B, L, V)
        # Gather token log-probs: Shape (B, L)
        gathered_log_probs: torch.Tensor = torch.gather(
            log_probs, dim=-1, index=targets.unsqueeze(-1)
        ).squeeze(-1)

        mask: torch.Tensor = (targets != self.ignore_index).float()
        return (gathered_log_probs * mask).sum(dim=-1)  # Shape: (B,)

    def forward(
        self,
        policy_chosen_logits: torch.Tensor,
        policy_rejected_logits: torch.Tensor,
        reference_chosen_logits: torch.Tensor,
        reference_rejected_logits: torch.Tensor,
        chosen_targets: torch.Tensor,
        rejected_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the scalar DPO loss over a batch of preference pairs."""
        if not (
            policy_chosen_logits.shape[0]
            == policy_rejected_logits.shape[0]
            == reference_chosen_logits.shape[0]
            == reference_rejected_logits.shape[0]
            == chosen_targets.shape[0]
            == rejected_targets.shape[0]
        ):
            raise TensorTopologyError("Batch dimensions of all DPO input tensors must match.")
        pi_chosen_logp: torch.Tensor = self._compute_sequence_log_probs(
            policy_chosen_logits, chosen_targets
        )
        pi_rejected_logp: torch.Tensor = self._compute_sequence_log_probs(
            policy_rejected_logits, rejected_targets
        )
        ref_chosen_logp: torch.Tensor = self._compute_sequence_log_probs(
            reference_chosen_logits, chosen_targets
        )
        ref_rejected_logp: torch.Tensor = self._compute_sequence_log_probs(
            reference_rejected_logits, rejected_targets
        )

        pi_ratio: torch.Tensor = pi_chosen_logp - pi_rejected_logp
        ref_ratio: torch.Tensor = ref_chosen_logp - ref_rejected_logp
        logits: torch.Tensor = self.beta * (pi_ratio - ref_ratio)

        loss: torch.Tensor = -F.logsigmoid(logits).mean()
        return loss
