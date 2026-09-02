"""
Conditional autoregressive generation: greedy and beam search decoding.

Supports expanded context window (L_max = 512 tokens) and robust multi-root
environment decoding ('tikzpicture', 'tikzcd', 'axis') with automatic package
inference and structural delimiter completion.

References:
    Goodfellow et al., Deep Learning — conditional sequence generation via the
        chain rule, teacher forcing (§10.2.1) and approximate MAP decoding (§12.4.3).
    Sutskever et al., Sequence to Sequence Learning with Neural Networks — beam
        search as approximate maximum-a-posteriori decoding over token sequences.
    Graves, Sequence Transduction with Recurrent Neural Networks — accumulated
        log-probability scoring and length normalization of beam hypotheses.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F

from core.dataset.packages import detect_required_packages
from core.exceptions import TensorTopologyError
from core.ml.model import VisionAutoregressiveModel
from core.models import (
    BOS_INDEX,
    EOS_INDEX,
    PAD_INDEX,
    ROOT_ENVIRONMENTS,
    UNK_INDEX,
    ImageTensor,
    TikzTokens,
    TokenVocabulary,
)

DEFAULT_MAX_SEQUENCE_LENGTH: int = 512
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


def _encode_single_image(model: VisionAutoregressiveModel, image: ImageTensor) -> torch.Tensor:
    """Return the visual memory ``(1, S, D)`` for a single-image batch."""
    if not isinstance(image, ImageTensor):
        raise TypeError("image must be an ImageTensor instance.")
    image_tensor: torch.Tensor = image.raw_tensor
    if image_tensor.ndim != 4:
        raise TensorTopologyError("Image must be a rank-4 tensor with shape (B, C, H, W).")
    if image_tensor.shape[0] != 1:
        raise TensorTopologyError("Inference requires an image batch of size one.")
    model_device: torch.device = next(model.parameters()).device
    if image_tensor.device != model_device:
        image_tensor = image_tensor.to(model_device)
    return cast(torch.Tensor, model.encoder(image_tensor))


def build_grammar_mask(
    vocabulary: TokenVocabulary,
    prefix_indices: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    """Compute a boolean mask over vocabulary tokens admissible under TikZ syntax rules.

    Admissibility conditions:
        - At step 0 (only BOS present), only root openers (\\begin{tikzpicture}) are allowed.
        - While delimiters ((), [], {}) are open, closing the root or emitting EOS is barred.
        - Delimiters cannot be closed if their current nesting depth is zero.
        - Delimiters cannot be opened if nested delimiter depth limits are reached.
        - Emitting \\end{...} requires balanced delimiters and a terminated statement (';').
        - Once \\end{...} is emitted, only EOS_INDEX is permitted.

    Args:
        vocabulary (TokenVocabulary): Discrete token vocabulary.
        prefix_indices (Sequence[int]): Previously emitted token indices starting with BOS_INDEX.
        device (torch.device): Execution device for the output boolean mask tensor.

    Returns:
        torch.Tensor: Boolean tensor of shape ``(V,)`` where True denotes admissible next tokens.

    Temporal complexity: O(L + V) where L is prefix length and V is vocabulary size.
    Spatial complexity: O(V).
    """
    vocab_size: int = len(vocabulary.token_to_index)
    mask: torch.Tensor = torch.ones(vocab_size, dtype=torch.bool, device=device)
    if PAD_INDEX in vocabulary.index_to_token:
        mask[PAD_INDEX] = False
    if UNK_INDEX in vocabulary.index_to_token:
        mask[UNK_INDEX] = False

    tokens: list[str] = [
        vocabulary.index_to_token[idx]
        for idx in prefix_indices
        if idx in vocabulary.index_to_token and idx not in (BOS_INDEX, PAD_INDEX, UNK_INDEX)
    ]

    if not tokens:
        mask.fill_(False)
        for env in ROOT_ENVIRONMENTS:
            begin_tag = f"\\begin{{{env}}}"
            if begin_tag in vocabulary.token_to_index:
                mask[vocabulary.token_to_index[begin_tag]] = True
        return mask

    paren_depth: int = 0
    bracket_depth: int = 0
    brace_depth: int = 0
    statements_completed: int = 0
    has_pending_statement: bool = False
    root_closed: bool = False

    for tok in tokens:
        if tok.startswith("\\end{"):
            root_closed = True
        elif tok == ";":
            statements_completed += 1
            has_pending_statement = False
        elif tok.startswith("\\") and tok not in (
            "\\x",
            "\\begin{tikzpicture}",
            "\\begin{axis}",
            "\\begin{tikzcd}",
        ):
            has_pending_statement = True
        elif tok == "(":
            paren_depth += 1
        elif tok == ")":
            paren_depth = max(0, paren_depth - 1)
        elif tok == "[":
            bracket_depth += 1
        elif tok == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif tok == "{":
            brace_depth += 1
        elif tok == "}":
            brace_depth = max(0, brace_depth - 1)

    if root_closed:
        mask.fill_(False)
        mask[EOS_INDEX] = True
        return mask

    mask[EOS_INDEX] = False

    for env in ROOT_ENVIRONMENTS:
        begin_tag = f"\\begin{{{env}}}"
        if begin_tag in vocabulary.token_to_index:
            mask[vocabulary.token_to_index[begin_tag]] = False

    if paren_depth == 0 and ")" in vocabulary.token_to_index:
        mask[vocabulary.token_to_index[")"]] = False
    if bracket_depth == 0 and "]" in vocabulary.token_to_index:
        mask[vocabulary.token_to_index["]"]] = False
    if brace_depth == 0 and "}" in vocabulary.token_to_index:
        mask[vocabulary.token_to_index["}"]] = False

    if paren_depth > 0:
        if "(" in vocabulary.token_to_index:
            mask[vocabulary.token_to_index["("]] = False
        if "[" in vocabulary.token_to_index:
            mask[vocabulary.token_to_index["["]] = False
        if ";" in vocabulary.token_to_index:
            mask[vocabulary.token_to_index[";"]] = False
        for env in ROOT_ENVIRONMENTS:
            end_tag = f"\\end{{{env}}}"
            if end_tag in vocabulary.token_to_index:
                mask[vocabulary.token_to_index[end_tag]] = False

    if bracket_depth > 0:
        if "[" in vocabulary.token_to_index:
            mask[vocabulary.token_to_index["["]] = False
        if ";" in vocabulary.token_to_index:
            mask[vocabulary.token_to_index[";"]] = False
        for env in ROOT_ENVIRONMENTS:
            end_tag = f"\\end{{{env}}}"
            if end_tag in vocabulary.token_to_index:
                mask[vocabulary.token_to_index[end_tag]] = False

    if brace_depth > 0:
        if ";" in vocabulary.token_to_index:
            mask[vocabulary.token_to_index[";"]] = False
        for env in ROOT_ENVIRONMENTS:
            end_tag = f"\\end{{{env}}}"
            if end_tag in vocabulary.token_to_index:
                mask[vocabulary.token_to_index[end_tag]] = False

    can_close_root: bool = (
        paren_depth == 0
        and bracket_depth == 0
        and brace_depth == 0
        and statements_completed > 0
        and not has_pending_statement
    )
    if not can_close_root:
        for env in ROOT_ENVIRONMENTS:
            end_tag = f"\\end{{{env}}}"
            if end_tag in vocabulary.token_to_index:
                mask[vocabulary.token_to_index[end_tag]] = False

    last_tok = tokens[-1] if tokens else ""
    if (
        paren_depth > 0 or bracket_depth > 0 or brace_depth > 0 or last_tok == ";"
    ) and ";" in vocabulary.token_to_index:
        mask[vocabulary.token_to_index[";"]] = False

    if not bool(mask.any().item()):
        mask.fill_(True)
        if PAD_INDEX in vocabulary.index_to_token:
            mask[PAD_INDEX] = False
        if UNK_INDEX in vocabulary.index_to_token:
            mask[UNK_INDEX] = False

    return mask


def greedy_search(
    model: VisionAutoregressiveModel,
    image: ImageTensor,
    max_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    grammar_constrained: bool = False,
) -> tuple[int, ...]:
    """Decode one image greedily into a ``[BOS, ..., EOS]`` token index sequence.

    At each step the token maximizing the conditional distribution is selected,
    and EOS-free columns are masked to ``EOS_INDEX`` once finished so the
    emitted sequence never extends past its terminating sentinel.

    Args:
        model (VisionAutoregressiveModel): Trained encoder/decoder in eval mode.
        image (ImageTensor): Single image with shape ``(1, C, H, W)``.
        max_length (int): Inclusive upper bound on sequence length. Default: 512.
        grammar_constrained (bool): If True, masks syntactically invalid tokens.

    Returns:
        tuple[int, ...]: Token index sequence including ``BOS_INDEX`` and, when
            emitted before truncation, ``EOS_INDEX``.

    Raises:
        TypeError: If model or image violate type specifications.
        ValueError: If max_length is invalid.

    Temporal complexity: O(L * T) where L is the emitted length and T is the
        causal decoder cost per step.
    """
    if not isinstance(model, VisionAutoregressiveModel):
        raise TypeError("model must be a VisionAutoregressiveModel instance.")
    if max_length < 2:
        raise ValueError(f"max_length must be at least 2. Got {max_length}.")
    if max_length > model.max_length:
        raise ValueError(f"max_length {max_length} exceeds model.max_length {model.max_length}.")

    visual_tokens: torch.Tensor = _encode_single_image(model, image)
    generated: torch.Tensor = torch.full(
        (1, 1), BOS_INDEX, dtype=torch.long, device=visual_tokens.device
    )
    finished: torch.Tensor = torch.zeros(1, dtype=torch.bool, device=visual_tokens.device)
    step: int = 0
    while step < max_length - 1 and not bool(finished.all().item()):
        logits: torch.Tensor = model.decoder(visual_tokens, generated)
        step_logits: torch.Tensor = logits[:, -1, :].clone()
        if grammar_constrained:
            current_seq: list[int] = generated[0].tolist()
            mask: torch.Tensor = build_grammar_mask(
                model.vocabulary, current_seq, device=visual_tokens.device
            )
            step_logits = step_logits.masked_fill(~mask, float("-inf"))

        next_token: torch.Tensor = step_logits.argmax(dim=-1)
        next_token = torch.where(finished, torch.full_like(next_token, EOS_INDEX), next_token)
        generated = torch.cat((generated, next_token.unsqueeze(1)), dim=1)
        finished = finished | next_token.eq(EOS_INDEX)
        step += 1

    return tuple(generated[0].tolist())


def sample_search(
    model: VisionAutoregressiveModel,
    image: ImageTensor,
    max_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    temperature: float = 0.7,
    top_p: float = 0.9,
    grammar_constrained: bool = True,
) -> tuple[int, ...]:
    """Sample one image into a token sequence using nucleus sampling and grammar constraints.

    Args:
        model (VisionAutoregressiveModel): Trained encoder/decoder in eval mode.
        image (ImageTensor): Single image with shape ``(1, C, H, W)``.
        max_length (int): Inclusive upper bound on sequence length. Default: 512.
        temperature (float): Softmax sampling temperature. Default: 0.7.
        top_p (float): Cumulative probability threshold for nucleus filtering. Default: 0.9.
        grammar_constrained (bool): If True, masks syntactically invalid tokens. Default: True.

    Returns:
        tuple[int, ...]: Emitted token sequence starting with BOS_INDEX.

    Temporal complexity: O(L * T) where L is emitted length and T is decoder step cost.
    """
    if not isinstance(model, VisionAutoregressiveModel):
        raise TypeError("model must be a VisionAutoregressiveModel instance.")
    if max_length < 2:
        raise ValueError(f"max_length must be at least 2. Got {max_length}.")
    if max_length > model.max_length:
        raise ValueError(f"max_length {max_length} exceeds model.max_length {model.max_length}.")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive. Got {temperature}.")
    if not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0.0, 1.0]. Got {top_p}.")

    visual_tokens: torch.Tensor = _encode_single_image(model, image)
    generated: torch.Tensor = torch.full(
        (1, 1), BOS_INDEX, dtype=torch.long, device=visual_tokens.device
    )
    finished: torch.Tensor = torch.zeros(1, dtype=torch.bool, device=visual_tokens.device)
    step: int = 0
    while step < max_length - 1 and not bool(finished.all().item()):
        logits: torch.Tensor = model.decoder(visual_tokens, generated)
        step_logits: torch.Tensor = logits[:, -1, :] / temperature

        if grammar_constrained:
            current_seq: list[int] = generated[0].tolist()
            mask: torch.Tensor = build_grammar_mask(
                model.vocabulary, current_seq, device=visual_tokens.device
            )
            step_logits = step_logits.masked_fill(~mask, float("-inf"))

        probs: torch.Tensor = F.softmax(step_logits, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=1, index=sorted_indices, src=sorted_indices_to_remove
            )
            probs = probs.masked_fill(indices_to_remove, 0.0)
            prob_sum = probs.sum(dim=-1, keepdim=True)
            probs = torch.where(prob_sum > 0, probs / prob_sum, probs)

        next_token: torch.Tensor = torch.multinomial(probs, num_samples=1).squeeze(1)
        next_token = torch.where(finished, torch.full_like(next_token, EOS_INDEX), next_token)
        generated = torch.cat((generated, next_token.unsqueeze(1)), dim=1)
        finished = finished | next_token.eq(EOS_INDEX)
        step += 1

    return tuple(generated[0].tolist())


def best_of_n_search(
    model: VisionAutoregressiveModel,
    image: ImageTensor,
    n_hypotheses: int = 4,
    max_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    temperature: float = 0.7,
    top_p: float = 0.9,
    grammar_constrained: bool = True,
) -> list[tuple[int, ...]]:
    """Generate N candidate token sequences: 1 greedy anchor + (N-1) nucleus-sampled candidates.

    Args:
        model (VisionAutoregressiveModel): Trained encoder/decoder in eval mode.
        image (ImageTensor): Single image with shape ``(1, C, H, W)``.
        n_hypotheses (int): Number of candidate sequences to produce (minimum 1).
        max_length (int): Inclusive upper bound on sequence length.
        temperature (float): Softmax sampling temperature for sampled candidates.
        top_p (float): Cumulative probability threshold for nucleus filtering.
        grammar_constrained (bool): If True, applies syntactic admissibility constraints.

    Returns:
        list[tuple[int, ...]]: Distinct candidate token index sequences.

    Temporal complexity: O(N * L * T) where N is hypothesis count.
    """
    if n_hypotheses < 1:
        raise ValueError(f"n_hypotheses must be at least 1. Got {n_hypotheses}.")

    greedy_candidate: tuple[int, ...] = greedy_search(
        model, image, max_length=max_length, grammar_constrained=grammar_constrained
    )
    candidates: list[tuple[int, ...]] = [greedy_candidate]

    sample_count: int = 1
    while sample_count < n_hypotheses:
        cand: tuple[int, ...] = sample_search(
            model,
            image,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            grammar_constrained=grammar_constrained,
        )
        candidates.append(cand)
        sample_count += 1

    return candidates


def beam_search(
    model: VisionAutoregressiveModel,
    image: ImageTensor,
    beam_width: int = 3,
    max_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    length_penalty: float = 0.0,
    grammar_constrained: bool = False,
) -> list[BeamHypothesis]:
    """Decode one image with beam search into ranked hypotheses.

    At each step every active beam is decoded in a single batch; the
    ``beam_width`` best ``(beam, token)`` candidates across the flattened
    ``(K, V)`` score matrix are kept. Candidates emitting ``EOS_INDEX`` are
    parked as completed hypotheses and never re-expanded.

    Args:
        model (VisionAutoregressiveModel): Trained encoder/decoder in eval mode.
        image (ImageTensor): Single image with shape ``(1, C, H, W)``.
        beam_width (int): Number of hypotheses retained at each step. Default: 3.
        max_length (int): Inclusive upper bound on sequence length. Default: 512.
        length_penalty (float): Non-negative exponent ``alpha``; hypotheses are
            ranked by ``log_probability / length ** alpha``. Default: 0.0.
        grammar_constrained (bool): If True, masks syntactically invalid tokens.

    Returns:
        list[BeamHypothesis]: Up to ``beam_width`` hypotheses sorted best-first.

    Raises:
        TypeError: If model or image violate type specifications.
        ValueError: If arguments violate structural boundaries.

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
        raise ValueError(f"max_length {max_length} exceeds model.max_length {model.max_length}.")
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
        step_logits: torch.Tensor = logits[:, -1, :].clone()

        if grammar_constrained:
            row_idx: int = 0
            while row_idx < len(active_sequences):
                mask: torch.Tensor = build_grammar_mask(
                    model.vocabulary, active_sequences[row_idx], device=visual_tokens.device
                )
                step_logits[row_idx, ~mask] = float("-inf")
                row_idx += 1

        log_probs: torch.Tensor = F.log_softmax(step_logits, dim=-1)
        score_matrix: torch.Tensor = log_probs + torch.tensor(
            active_scores, device=visual_tokens.device
        ).unsqueeze(1)
        flat_scores: torch.Tensor = score_matrix.reshape(-1)
        top_scores, top_indices = flat_scores.topk(min(beam_width, flat_scores.numel()))

        next_sequences: list[list[int]] = []
        next_scores: list[float] = []
        for rank_score, flat_index in zip(top_scores.tolist(), top_indices.tolist(), strict=True):
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
        key=lambda hypothesis: (
            hypothesis.log_probability / (len(hypothesis.tokens) ** length_penalty)
        ),
        reverse=True,
    )
    return ranked[:beam_width]



def _reconstruct_environment_markup(
    decoded_tokens: list[str],
) -> tuple[str, tuple[str, ...]]:
    """
    Reconstructs markup string and detects package dependencies for multi-root environments.

    Preserves root environments ('tikzpicture', 'tikzcd', 'axis') with automatic delimiter
    completion for truncated sequences, and wraps bare drawing commands in 'tikzpicture'.

    Args:
        decoded_tokens (list[str]): Decoded token strings excluding special sentinels.

    Returns:
        tuple[str, tuple[str, ...]]: (reconstructed_markup, detected_packages).

    Temporal complexity: O(L) where L is token length.
    """
    if not decoded_tokens:
        markup: str = f"{_BEGIN_TIKZ} {_END_TIKZ}"
        return markup, ()

    # Identify declared root environment if present
    root_env: str | None = None
    for token in decoded_tokens:
        if root_env is None:
            for env_name in ROOT_ENVIRONMENTS:
                if token == f"\\begin{{{env_name}}}":
                    root_env = env_name

    final_tokens: list[str] = list(decoded_tokens)
    if root_env is not None:
        begin_tag: str = f"\\begin{{{root_env}}}"
        end_tag: str = f"\\end{{{root_env}}}"

        # Ensure begin tag is at sequence start
        if final_tokens[0] != begin_tag:
            final_tokens = [t for t in final_tokens if t != begin_tag]
            final_tokens = [begin_tag] + final_tokens

        # Ensure matching end tag terminates sequence
        if final_tokens[-1] != end_tag:
            final_tokens = [t for t in final_tokens if t != end_tag] + [end_tag]
    else:
        # Wrap bare drawing content in canonical tikzpicture delimiters
        filtered_content: list[str] = [t for t in final_tokens if t not in (_BEGIN_TIKZ, _END_TIKZ)]
        final_tokens = [_BEGIN_TIKZ, *filtered_content, _END_TIKZ]

    reconstructed_markup: str = " ".join(final_tokens)
    detected_packages: tuple[str, ...] = detect_required_packages(reconstructed_markup)
    return reconstructed_markup, detected_packages


def decode_indices_to_markup(vocabulary: TokenVocabulary, indices: tuple[int, ...]) -> TikzTokens:
    """
    Map a generated token index sequence onto a validated ``TikzTokens`` value object.

    Special sentinels (PAD, BOS, EOS, UNK) are discarded and multi-root environment
    delimiters ('tikzpicture', 'tikzcd', 'axis') are balanced and package-inferred.

    Args:
        vocabulary (TokenVocabulary): Index-to-token mapping used for decoding.
        indices (tuple[int, ...]): Generated sequence of integer token indices.

    Returns:
        TikzTokens: Immutable markup wrapped in a valid root environment with packages.

    Raises:
        TypeError: If vocabulary is not a TokenVocabulary instance.
        TensorTopologyError: If indices is not a valid tuple of integers.

    Temporal complexity: O(L) where L is the emitted sequence length.
    """
    if not isinstance(vocabulary, TokenVocabulary):
        raise TypeError("vocabulary must be a TokenVocabulary instance.")
    if not isinstance(indices, tuple) or not all(isinstance(index, int) for index in indices):
        raise TensorTopologyError("indices must be a tuple of integer token indices.")

    special_indices: frozenset[int] = frozenset({BOS_INDEX, EOS_INDEX, PAD_INDEX, UNK_INDEX})
    decoded_tokens: list[str] = [
        vocabulary.index_to_token[index]
        for index in indices
        if index not in special_indices and index in vocabulary.index_to_token
    ]

    markup, packages = _reconstruct_environment_markup(decoded_tokens)
    return TikzTokens(markup=markup, packages=packages)
