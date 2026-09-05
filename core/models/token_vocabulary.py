from dataclasses import dataclass

from core.exceptions import VocabularyInvariantError

PAD_TOKEN: str = "<PAD>"
BOS_TOKEN: str = "<BOS>"
EOS_TOKEN: str = "<EOS>"
UNK_TOKEN: str = "<UNK>"

PAD_INDEX: int = 0
BOS_INDEX: int = 1
EOS_INDEX: int = 2
UNK_INDEX: int = 3

FAMILY_NAMES: tuple[str, ...] = (
    "line_segment",
    "polyline",
    "polygon",
    "circle_arc",
    "grid_axes",
    "function_plot",
    "node_arrow",
    "composed",
)
FAMILY_PREFIX_TOKENS: tuple[str, ...] = tuple(f"<FAM:{fam}>" for fam in FAMILY_NAMES)


@dataclass(frozen=True)
class TokenVocabulary:
    """
    Bidirectional mapping between tokens and integer indices.

    Invariants:
        - |token_to_index| == |index_to_token|
        - Reserved tokens occupy fixed indices [0, 3]: PAD=0, BOS=1, EOS=2, UNK=3
        - All keys in token_to_index are non-empty strings

    Spatial complexity: O(|V|) where |V| is the vocabulary size.
    """

    token_to_index: dict[str, int]
    index_to_token: dict[int, str]

    def __post_init__(self) -> None:
        if not isinstance(self.token_to_index, dict) or not isinstance(self.index_to_token, dict):
            raise VocabularyInvariantError("Mappings must be dictionary instances.")

        if len(self.token_to_index) != len(self.index_to_token):
            raise VocabularyInvariantError(
                "Vocabulary mappings violate bijectivity cardinality equality."
            )

        # Validate reserved tokens
        reserved_tokens: list[tuple[str, int]] = [
            (PAD_TOKEN, PAD_INDEX),
            (BOS_TOKEN, BOS_INDEX),
            (EOS_TOKEN, EOS_INDEX),
            (UNK_TOKEN, UNK_INDEX),
        ]

        for token, idx in reserved_tokens:
            if self.token_to_index.get(token) != idx:
                raise VocabularyInvariantError(f"Reserved token '{token}' must map to index {idx}.")
            if self.index_to_token.get(idx) != token:
                raise VocabularyInvariantError(f"Index {idx} must map to reserved token '{token}'.")

        # Validate non-empty string keys
        for token in self.token_to_index:
            if not isinstance(token, str) or not token:
                raise VocabularyInvariantError("Vocabulary tokens must be non-empty strings.")
