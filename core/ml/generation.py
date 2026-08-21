"""
Conditional autoregressive generation: greedy and beam search decoding.

References:
    Goodfellow et al., Deep Learning — conditional sequence generation via the
        chain rule, teacher forcing (§10.2.1) and approximate MAP decoding (§12.4.3).
    Sutskever et al., Sequence to Sequence Learning with Neural Networks — beam
        search as approximate maximum-a-posteriori decoding over token sequences.
    Graves, Sequence Transduction with Recurrent Neural Networks — accumulated
        log-probability scoring and length normalization of beam hypotheses.
"""

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F

from core.exceptions import TensorTopologyError
from core.ml.model import VisionAutoregressiveModel
from core.models import (
    BOS_INDEX,
    EOS_INDEX,
    PAD_INDEX,
    UNK_INDEX,
    ImageTensor,
    TikzTokens,
    TokenVocabulary,
)

_BEGIN_TIKZ: str = r"\begin{tikzpicture}"
_END_TIKZ: str = r"\end{tikzpicture}"


@dataclass(frozen=True)
class BeamHypothesis:
    """A decoded sequence hypothesis with its accumulated log-probability."""

    tokens: tuple[int, ...]
    log_probability: float

    def __post_init__(self) -> None:
        if not self.tokens:
            raise TensorTopologyError("A beam hypothesis must contain at least one token.")
        if not all(isinstance(index, int) for index in self.tokens):
            raise TensorTopologyError("Beam hypothesis tokens must be integer indices.")


def _encode_single_image(
    model: VisionAutoregressiveModel, image: ImageTensor
) -> torch.Tensor:
    """Return the visual memory ``(1, S, D)`` for a single-image batch."""
    if not isinstance(image, ImageTensor):
        raise TypeError("image must be an ImageTensor instance.")
    image_tensor: torch.Tensor = image.raw_tensor
    if image_tensor.ndim != 4:
        raise TensorTopologyError("Image must be a rank-4 tensor with shape (B, C, H, W).")
    if image_tensor.shape[0] != 1:
        raise TensorTopologyError("Inference requires an image batch of size one.")
    return cast(torch.Tensor, model.encoder(image_tensor))


def greedy_search(
    model: VisionAutoregressiveModel,
    image: ImageTensor,
    max_length: int,
) -> tuple[int, ...]:
    """
    Decode one image greedily into a ``[BOS, ..., EOS]`` token index sequence.

    At each step the token maximizing the conditional distribution is selected,
    and EOS-free columns are masked to ``EOS_INDEX`` once finished so the
    emitted sequence never extends past its terminating sentinel.

    Args:
        model (VisionAutoregressiveModel): Trained encoder/decoder in eval mode.
        image (ImageTensor): Single image with shape ``(1, C, H, W)``.
        max_length (int): Inclusive upper bound on the emitted sequence length.

    Returns:
        tuple[int, ...]: Token index sequence including ``BOS_INDEX`` and, when
            emitted before truncation, ``EOS_INDEX``.

    Temporal complexity: O(L * T) where L is the emitted length and T is the
        causal decoder cost per step.
    """
    if not isinstance(model, VisionAutoregressiveModel):
        raise TypeError("model must be a VisionAutoregressiveModel instance.")
    if max_length < 2:
        raise ValueError(f"max_length must be at least 2. Got {max_length}.")
    if max_length > model.max_length:
        raise ValueError(
            f"max_length {max_length} exceeds model.max_length {model.max_length}."
        )

    visual_tokens: torch.Tensor = _encode_single_image(model, image)
    generated: torch.Tensor = torch.full(
        (1, 1), BOS_INDEX, dtype=torch.long, device=visual_tokens.device
    )
    finished: torch.Tensor = torch.zeros(1, dtype=torch.bool, device=visual_tokens.device)
    step: int = 0
    while step < max_length - 1 and not bool(finished.all().item()):
        logits: torch.Tensor = model.decoder(visual_tokens, generated)
        next_token: torch.Tensor = logits[:, -1, :].argmax(dim=-1)
        next_token = torch.where(
            finished, torch.full_like(next_token, EOS_INDEX), next_token
        )
        generated = torch.cat((generated, next_token.unsqueeze(1)), dim=1)
        finished = finished | next_token.eq(EOS_INDEX)
        step += 1

    return tuple(generated[0].tolist())


def beam_search(
    model: VisionAutoregressiveModel,
    image: ImageTensor,
    beam_width: int,
    max_length: int,
    length_penalty: float = 0.0,
) -> list[BeamHypothesis]:
    """
    Decode one image with beam search into ranked hypotheses.

    At each step every active beam is decoded in a single batch; the
    ``beam_width`` best ``(beam, token)`` candidates across the flattened
    ``(K, V)`` score matrix are kept. Candidates emitting ``EOS_INDEX`` are
    parked as completed hypotheses and never re-expanded.

    Args:
        model (VisionAutoregressiveModel): Trained encoder/decoder in eval mode.
        image (ImageTensor): Single image with shape ``(1, C, H, W)``.
        beam_width (int): Number of hypotheses retained at each step.
        max_length (int): Inclusive upper bound on the emitted sequence length.
        length_penalty (float): Non-negative exponent ``alpha``; hypotheses are
            ranked by ``log_probability / length ** alpha``.

    Returns:
        list[BeamHypothesis]: Up to ``beam_width`` hypotheses sorted best-first.

    Temporal complexity: O(L * B * (T + V)) where L is the emitted length, B the
        beam width, V the vocabulary size and T the decoder cost per step.
    """
    if not isinstance(model, VisionAutoregressiveModel):
        raise TypeError("model must be a VisionAutoregressiveModel instance.")
    if beam_width < 1:
        raise ValueError(f"beam_width must be at least 1. Got {beam_width}.")
    if max_length < 2:
        raise ValueError(f"max_length must be at least 2. Got {max_length}.")
    if max_length > model.max_length:
        raise ValueError(
            f"max_length {max_length} exceeds model.max_length {model.max_length}."
        )
    if length_penalty < 0.0:
        raise ValueError(f"length_penalty must be non-negative. Got {length_penalty}.")

    visual_tokens: torch.Tensor = _encode_single_image(model, image)
    vocabulary_size: int = len(model.vocabulary.token_to_index)

    active_sequences: list[list[int]] = [[BOS_INDEX]]
    active_scores: list[float] = [0.0]
    completed: list[BeamHypothesis] = []

    step: int = 0
    while step < max_length - 1 and active_sequences:
        batch: torch.Tensor = torch.tensor(
            active_sequences, dtype=torch.long, device=visual_tokens.device
        )
        expanded_memory: torch.Tensor = visual_tokens.expand(batch.shape[0], -1, -1)
        logits: torch.Tensor = model.decoder(expanded_memory, batch)
        log_probs: torch.Tensor = F.log_softmax(logits[:, -1, :], dim=-1)
        score_matrix: torch.Tensor = log_probs + torch.tensor(
            active_scores, device=visual_tokens.device
        ).unsqueeze(1)
        flat_scores: torch.Tensor = score_matrix.reshape(-1)
        top_scores, top_indices = flat_scores.topk(min(beam_width, flat_scores.numel()))

        next_sequences: list[list[int]] = []
        next_scores: list[float] = []
        for rank_score, flat_index in zip(
            top_scores.tolist(), top_indices.tolist(), strict=True
        ):
            parent: int = flat_index // vocabulary_size
            token: int = flat_index % vocabulary_size
            candidate: list[int] = active_sequences[parent] + [token]
            if token == EOS_INDEX:
                completed.append(
                    BeamHypothesis(tokens=tuple(candidate), log_probability=rank_score)
                )
            else:
                next_sequences.append(candidate)
                next_scores.append(rank_score)

        active_sequences = next_sequences
        active_scores = next_scores
        step += 1

    truncated: list[BeamHypothesis] = [
        BeamHypothesis(tokens=tuple(sequence), log_probability=score)
        for sequence, score in zip(active_sequences, active_scores, strict=True)
    ]
    ranked: list[BeamHypothesis] = sorted(
        completed + truncated,
        key=lambda hypothesis: hypothesis.log_probability
        / (len(hypothesis.tokens) ** length_penalty),
        reverse=True,
    )
    return ranked[:beam_width]


def decode_indices_to_markup(
    vocabulary: TokenVocabulary, indices: tuple[int, ...]
) -> TikzTokens:
    """
    Map a generated token index sequence onto ``TikzTokens`` markup.

    Special sentinels (PAD, BOS, EOS, UNK) are discarded and the tikzpicture
    environment delimiters are re-anchored around the decoded content.

    Args:
        vocabulary (TokenVocabulary): Index-to-token mapping used for decoding.
        indices (tuple[int, ...]): Generated sequence of integer token indices.

    Returns:
        TikzTokens: Immutable markup wrapped in a tikzpicture environment.

    Temporal complexity: O(L) where L is the emitted sequence length.
    """
    if not isinstance(vocabulary, TokenVocabulary):
        raise TypeError("vocabulary must be a TokenVocabulary instance.")
    if not isinstance(indices, tuple) or not all(
        isinstance(index, int) for index in indices
    ):
        raise TensorTopologyError("indices must be a tuple of integer token indices.")

    special_indices: frozenset[int] = frozenset(
        {BOS_INDEX, EOS_INDEX, PAD_INDEX, UNK_INDEX}
    )
    decoded_tokens: list[str] = [
        vocabulary.index_to_token[index]
        for index in indices
        if index not in special_indices and index in vocabulary.index_to_token
    ]
    content_tokens: list[str] = [
        token for token in decoded_tokens if token not in (_BEGIN_TIKZ, _END_TIKZ)
    ]
    return TikzTokens(markup=" ".join([_BEGIN_TIKZ, *content_tokens, _END_TIKZ]))
