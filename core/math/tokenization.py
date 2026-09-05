"""
Bidirectional tokenization primitives (markup text <-> integer indices).

Provides configurable coordinate quantization over the continuous canvas [-5, 5]^2.
The V3 profile uses 0.05-spaced bins (201 values per axis), compacting the
vocabulary from unbounded coordinate combinations to a bounded token space.

References:
    Golub & Van Loan, Matrix Computations — discrete quantization, uniform
        coordinate binning, and integer index projections.
    Goodfellow et al., Deep Learning — autoregressive sequence modeling,
        vocabulary construction, and token embedding layers.
"""

import re

import torch

from core.exceptions import VocabularyInvariantError
from core.models.token_vocabulary import (
    BOS_INDEX,
    BOS_TOKEN,
    EOS_INDEX,
    EOS_TOKEN,
    FAMILY_PREFIX_TOKENS,
    PAD_INDEX,
    PAD_TOKEN,
    UNK_INDEX,
    UNK_TOKEN,
    TokenVocabulary,
)
from core.models.value_objects import TikzTokens

# Canonical canvas spatial domain constants
CANVAS_MIN: float = -5.0
CANVAS_MAX: float = 5.0
COORDINATE_STEP: float = 0.1
NUM_COORDINATE_BINS: int = 100
V3_COORDINATE_STEP: float = 0.05
V3_NUM_COORDINATE_BINS: int = 200

# Pre-computed spatial coordinate bins: 101 discrete points spanning [-5.0, 5.0]
COORDINATE_BINS: tuple[str, ...] = tuple(
    f"{round(CANVAS_MIN + idx * COORDINATE_STEP, 1):g}" for idx in range(NUM_COORDINATE_BINS + 1)
)
V3_COORDINATE_BINS: tuple[str, ...] = tuple(
    f"{round(CANVAS_MIN + idx * V3_COORDINATE_STEP, 2):g}"
    for idx in range(V3_NUM_COORDINATE_BINS + 1)
)

# Regex pattern matching discrete TikZ tokens, environments, operators, and identifiers
TIKZ_TOKEN_PATTERN: re.Pattern[str] = re.compile(
    r"<FAM:[a-zA-Z0-9_]+>"
    r"|\\begin\{[a-zA-Z*]+\}"
    r"|\\end\{[a-zA-Z*]+\}"
    r"|\\[a-zA-Z]+"
    r"|--|->|<-|<->|\|-|-\||\.\.|\+\+|->>|-stealth"
    r"|-?\d+(?:\.\d+)?"
    r"|[a-zA-Z_][a-zA-Z0-9_-]*"
    r"|[^\s]"
)

_FLOAT_PATTERN: re.Pattern[str] = re.compile(r"^-?\d+\.\d+$")


def quantize_coordinate_scalar(
    val: float,
    min_val: float = CANVAS_MIN,
    max_val: float = CANVAS_MAX,
    step: float = COORDINATE_STEP,
) -> float:
    """
    Uniform scalar coordinate binning on [min_val, max_val] with discrete step.

    Applies clamping and round-to-nearest projection onto the 1D lattice:
        q(x) = min_val + round((clamp(x, min_val, max_val) - min_val) / step) * step

    Args:
        val (float): Real-valued input coordinate.
        min_val (float): Lower canvas boundary. Default: -5.0.
        max_val (float): Upper canvas boundary. Default: 5.0.
        step (float): Discretization step. Default: 0.1.

    Returns:
        float: Discretized, clamped scalar coordinate.

    Raises:
        VocabularyInvariantError: On invalid domain boundaries or non-positive step.
        TypeError: If input values are not numeric.

    Temporal complexity: O(1).
    """
    if not isinstance(val, (int, float)):
        raise TypeError(f"Coordinate scalar must be numeric. Got {type(val).__name__}.")
    if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
        raise TypeError("Domain boundaries must be numeric.")
    if not isinstance(step, (int, float)):
        raise TypeError("Discretization step must be numeric.")
    if min_val >= max_val:
        raise VocabularyInvariantError(
            f"min_val ({min_val}) must be strictly less than max_val ({max_val})."
        )
    if step <= 0.0:
        raise VocabularyInvariantError(f"step must be strictly positive. Got {step}.")

    clamped: float = max(float(min_val), min(float(max_val), float(val)))
    bin_idx: int = round((clamped - min_val) / step)
    quantized: float = round(min_val + bin_idx * step, 4)
    normalized: float = 0.0 if abs(quantized) < 1e-9 else quantized
    return normalized


def quantize_coordinate_tuple(
    point: tuple[float, float],
    min_val: float = CANVAS_MIN,
    max_val: float = CANVAS_MAX,
    step: float = COORDINATE_STEP,
) -> tuple[float, float]:
    """
    Quantizes a 2D coordinate point onto the uniform grid [min_val, max_val]^2.

    Args:
        point (tuple[float, float]): Input coordinate (x, y).
        min_val (float): Lower canvas boundary. Default: -5.0.
        max_val (float): Upper canvas boundary. Default: 5.0.
        step (float): Discretization step. Default: 0.1.

    Returns:
        tuple[float, float]: Quantized (qx, qy) tuple.

    Temporal complexity: O(1).
    """
    if not isinstance(point, tuple) or len(point) != 2:
        raise TypeError("Point must be a 2-element tuple of floats.")

    return (
        quantize_coordinate_scalar(point[0], min_val=min_val, max_val=max_val, step=step),
        quantize_coordinate_scalar(point[1], min_val=min_val, max_val=max_val, step=step),
    )


def tokenize_tikz_markup(
    tokens: TikzTokens,
    quantize: bool = True,
    coordinate_step: float = COORDINATE_STEP,
) -> list[str]:
    """
    Splits TikZ markup into discrete tokens with coordinate quantization.

    Args:
        tokens (TikzTokens): Input markup value object.
        quantize (bool): When True, discretizes real-valued floating literals
            onto the configured canvas grid. Default: True.

    Returns:
        list[str]: Extracted token strings.

    Raises:
        TypeError: If input is not a TikzTokens instance.

    Temporal complexity: O(N) where N is the length of the markup string.
    """
    if not isinstance(tokens, TikzTokens):
        raise TypeError("Input must be a TikzTokens instance.")

    raw_tokens: list[str] = TIKZ_TOKEN_PATTERN.findall(tokens.markup)
    if not quantize:
        return raw_tokens

    processed_tokens: list[str] = []
    for token in raw_tokens:
        if _FLOAT_PATTERN.fullmatch(token):
            quantized_val: float = quantize_coordinate_scalar(float(token), step=coordinate_step)
            processed_tokens.append(f"{quantized_val:g}")
        else:
            processed_tokens.append(token)

    return processed_tokens


def build_vocabulary(
    corpus: list[TikzTokens],
    include_spatial_grid: bool = True,
    quantize: bool = True,
    coordinate_step: float = COORDINATE_STEP,
) -> TokenVocabulary:
    """
    Constructs the TokenVocabulary from a TikZ corpus with bounded spatial support.

    Args:
        corpus (list[TikzTokens]): Sequence of TikZ document samples.
        include_spatial_grid (bool): When True, pre-populates all 101 spatial
            coordinate bins in [-5.0, 5.0] to ensure complete coverage. Default: True.
        quantize (bool): When True, quantizes coordinates during extraction. Default: True.

    Returns:
        TokenVocabulary: Invariant-enforced bidirectional token vocabulary.

    Raises:
        TypeError: If corpus is not a list.

    Temporal complexity: O(|C| * |T|) where |C| is corpus size and |T| is token length.
    """
    if not isinstance(corpus, list):
        raise TypeError("Corpus must be a list of TikzTokens instances.")

    all_token_lists: list[list[str]] = [
        tokenize_tikz_markup(doc, quantize=quantize, coordinate_step=coordinate_step)
        for doc in corpus
    ]
    flat_tokens: set[str] = {token for sublist in all_token_lists for token in sublist}

    if include_spatial_grid:
        bin_count = round((CANVAS_MAX - CANVAS_MIN) / coordinate_step)
        flat_tokens.update(
            f"{round(CANVAS_MIN + idx * coordinate_step, 4):g}"
            for idx in range(bin_count + 1)
        )

    flat_tokens.update(FAMILY_PREFIX_TOKENS)

    reserved_set: set[str] = {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN}
    unique_tokens: list[str] = sorted(flat_tokens - reserved_set)

    token_to_index: dict[str, int] = {
        PAD_TOKEN: PAD_INDEX,
        BOS_TOKEN: BOS_INDEX,
        EOS_TOKEN: EOS_INDEX,
        UNK_TOKEN: UNK_INDEX,
    }
    index_to_token: dict[int, str] = {
        PAD_INDEX: PAD_TOKEN,
        BOS_INDEX: BOS_TOKEN,
        EOS_INDEX: EOS_TOKEN,
        UNK_INDEX: UNK_TOKEN,
    }

    for idx, token in enumerate(unique_tokens, start=4):
        token_to_index[token] = idx
        index_to_token[idx] = token

    return TokenVocabulary(token_to_index=token_to_index, index_to_token=index_to_token)


def encode_to_tensor(
    tokens: TikzTokens,
    vocabulary: TokenVocabulary,
    max_length: int = 512,
    quantize: bool = True,
    coordinate_step: float = COORDINATE_STEP,
) -> torch.Tensor:
    """
    Encodes markup tokens into a rank-1 integer index tensor.

    Args:
        tokens (TikzTokens): Input markup.
        vocabulary (TokenVocabulary): Bidirectional token vocabulary.
        max_length (int): Fixed output sequence length. Default: 512.
        quantize (bool): Whether to quantize continuous coordinates. Default: True.

    Returns:
        torch.Tensor: Rank-1 long index tensor of shape (max_length,).

    Raises:
        VocabularyInvariantError: If max_length is non-positive.
        TypeError: If inputs fail type constraints.

    Temporal complexity: O(T) where T is the token sequence length.
    """
    if max_length <= 0:
        raise VocabularyInvariantError(f"max_length must be positive. Got {max_length}.")

    if not isinstance(tokens, TikzTokens):
        raise TypeError("Input tokens must be a TikzTokens instance.")

    if not isinstance(vocabulary, TokenVocabulary):
        raise TypeError("Vocabulary must be a TokenVocabulary instance.")

    string_tokens: list[str] = tokenize_tikz_markup(
        tokens, quantize=quantize, coordinate_step=coordinate_step
    )
    token_indices: list[int] = (
        [BOS_INDEX]
        + [vocabulary.token_to_index.get(token, UNK_INDEX) for token in string_tokens]
        + [EOS_INDEX]
    )

    sequence_len: int = len(token_indices)
    if sequence_len > max_length:
        final_indices: list[int] = token_indices[:max_length]
    else:
        final_indices = token_indices + [PAD_INDEX] * (max_length - sequence_len)

    # Shape: (max_length,)
    encoded_tensor: torch.Tensor = torch.tensor(final_indices, dtype=torch.long)
    return encoded_tensor


def decode_from_tensor(tensor: torch.Tensor, vocabulary: TokenVocabulary) -> TikzTokens:
    """
    Decodes a rank-1 integer index tensor back into a validated TikzTokens entity.

    Args:
        tensor (torch.Tensor): Rank-1 index tensor of shape (max_length,).
        vocabulary (TokenVocabulary): Bidirectional token vocabulary.

    Returns:
        TikzTokens: Reconstructed markup value object.

    Raises:
        TypeError: If inputs fail type validation.
        VocabularyInvariantError: If tensor rank is not 1.

    Temporal complexity: O(T) where T is the sequence length.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor instance.")

    if not isinstance(vocabulary, TokenVocabulary):
        raise TypeError("Vocabulary must be a TokenVocabulary instance.")

    if tensor.ndim != 1:
        raise VocabularyInvariantError(f"Expected 1D tensor for decoding, got {tensor.ndim}D.")

    indices: list[int] = tensor.tolist()
    ignored_indices: set[int] = {PAD_INDEX, BOS_INDEX, EOS_INDEX}

    extracted_tokens: list[str] = [
        vocabulary.index_to_token.get(idx, UNK_TOKEN)
        for idx in indices
        if idx not in ignored_indices
    ]

    reconstructed_markup: str = " ".join(extracted_tokens)
    return TikzTokens(markup=reconstructed_markup)


def batch_encode(
    corpus: list[TikzTokens],
    vocabulary: TokenVocabulary,
    max_length: int = 512,
    quantize: bool = True,
    coordinate_step: float = COORDINATE_STEP,
) -> torch.Tensor:
    """
    Encodes a batch of TikZ documents into a 2D integer index tensor.

    Args:
        corpus (list[TikzTokens]): Input documents.
        vocabulary (TokenVocabulary): Bidirectional token vocabulary.
        max_length (int): Fixed sequence length per sample. Default: 512.
        quantize (bool): Whether to quantize continuous coordinates. Default: True.

    Returns:
        torch.Tensor: Batch tensor of shape (N, max_length).

    Raises:
        TypeError: If corpus is not a list.
        VocabularyInvariantError: If corpus is empty.

    Temporal complexity: O(N * T) where N is batch size and T is max_length.
    """
    if not isinstance(corpus, list):
        raise TypeError("Corpus must be a list of TikzTokens instances.")

    if not corpus:
        raise VocabularyInvariantError("Corpus cannot be empty for batch encoding.")

    tensors: list[torch.Tensor] = [
        encode_to_tensor(
            doc,
            vocabulary,
            max_length=max_length,
            quantize=quantize,
            coordinate_step=coordinate_step,
        )
        for doc in corpus
    ]

    # Shape: (N, max_length)
    batch_tensor: torch.Tensor = torch.stack(tensors, dim=0)
    return batch_tensor
