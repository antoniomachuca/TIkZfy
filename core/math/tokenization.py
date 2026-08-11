"""
Bidirectional tokenization primitives (markup text <-> integer indices).

References: Golub & Van Loan, Matrix Computations (integer indexing);
Goodfellow et al., Deep Learning (autoregressive sequence modeling).
"""
import re

import torch

from core.exceptions import VocabularyInvariantError
from core.models.token_vocabulary import (
    BOS_INDEX,
    BOS_TOKEN,
    EOS_INDEX,
    EOS_TOKEN,
    PAD_INDEX,
    PAD_TOKEN,
    UNK_INDEX,
    UNK_TOKEN,
    TokenVocabulary,
)
from core.models.value_objects import TikzTokens

# Regex pattern matching TikZ tokens
TIKZ_TOKEN_PATTERN: re.Pattern[str] = re.compile(
    r"\\begin\{tikzpicture\}"
    r"|\\end\{tikzpicture\}"
    r"|\\[a-zA-Z]+"
    r"|\(-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\)"
    r"|--|->|<-|<->|\|-|-\||\.\."
    r"|-?\d+(?:\.\d+)?"
    r"|[a-zA-Z_][a-zA-Z0-9_-]*"
    r"|[^\s]"
)



def tokenize_tikz_markup(tokens: TikzTokens) -> list[str]:
    """
    Splits TikZ markup into individual tokens.

    Args:
        tokens (TikzTokens): Input markup.

    Returns:
        list[str]: The extracted tokens.

    Temporal complexity: O(N) where N is the length of the markup.
    """
    if not isinstance(tokens, TikzTokens):
        raise TypeError("Input must be a TikzTokens instance.")

    return TIKZ_TOKEN_PATTERN.findall(tokens.markup)


def build_vocabulary(corpus: list[TikzTokens]) -> TokenVocabulary:
    """
    Constructs the TokenVocabulary from a TikZ corpus.

    Args:
        corpus (list[TikzTokens]): Sequence of TikZ document samples.

    Returns:
        TokenVocabulary: Vocabulary entity with reserved indices.

    Temporal complexity: O(|C| * |T|) where |C| is corpus size and |T| is token length.
    """
    if not isinstance(corpus, list):
        raise TypeError("Corpus must be a list of TikzTokens instances.")

    # Extract tokens for all documents without explicit loops
    all_token_lists: list[list[str]] = list(map(tokenize_tikz_markup, corpus))
    flat_tokens: set[str] = {token for sublist in all_token_lists for token in sublist}

    # Filter reserved tokens to prevent collisions
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

    # Assign sequential indices to remaining unique tokens starting at index 4
    for idx, token in enumerate(unique_tokens, start=4):
        token_to_index[token] = idx
        index_to_token[idx] = token

    return TokenVocabulary(token_to_index=token_to_index, index_to_token=index_to_token)


def encode_to_tensor(
    tokens: TikzTokens,
    vocabulary: TokenVocabulary,
    max_length: int = 512,
) -> torch.Tensor:
    """
    Encodes markup tokens into an integer tensor.

    Args:
        tokens (TikzTokens): Input markup.
        vocabulary (TokenVocabulary): Token vocabulary.
        max_length (int): Fixed output sequence length. Default: 512.

    Returns:
        torch.Tensor: Rank-1 index tensor. Shape: (max_length,)

    Temporal complexity: O(T) where T is the sequence length.
    """
    if max_length <= 0:
        raise VocabularyInvariantError(f"max_length must be positive. Got {max_length}.")

    if not isinstance(tokens, TikzTokens):
        raise TypeError("Input tokens must be a TikzTokens instance.")

    if not isinstance(vocabulary, TokenVocabulary):
        raise TypeError("Vocabulary must be a TokenVocabulary instance.")

    string_tokens: list[str] = tokenize_tikz_markup(tokens)
    token_indices: list[int] = [BOS_INDEX] + [
        vocabulary.token_to_index.get(token, UNK_INDEX) for token in string_tokens
    ] + [EOS_INDEX]

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
    Decodes an integer index tensor back into markup.

    Args:
        tensor (torch.Tensor): Rank-1 index tensor. Shape: (max_length,)
        vocabulary (TokenVocabulary): Token vocabulary.

    Returns:
        TikzTokens: The reconstructed markup.

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
) -> torch.Tensor:
    """
    Encodes a batch of TikZ documents into a 2D index tensor.

    Args:
        corpus (list[TikzTokens]): Input documents.
        vocabulary (TokenVocabulary): Token vocabulary.
        max_length (int): Fixed sequence length per sample. Default: 512.

    Returns:
        torch.Tensor: Batch tensor. Shape: (N, max_length)

    Temporal complexity: O(N * T) where N is batch size and T is max sequence length.
    """
    if not isinstance(corpus, list):
        raise TypeError("Corpus must be a list of TikzTokens instances.")

    if not corpus:
        raise VocabularyInvariantError("Corpus cannot be empty for batch encoding.")

    tensors: list[torch.Tensor] = [
        encode_to_tensor(doc, vocabulary, max_length=max_length) for doc in corpus
    ]

    # Shape: (N, max_length)
    batch_tensor: torch.Tensor = torch.stack(tensors, dim=0)
    return batch_tensor
