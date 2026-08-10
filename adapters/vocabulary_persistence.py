import json
from pathlib import Path

from core.exceptions import DomainError, VocabularyInvariantError
from core.models import TokenVocabulary
from ports.outbound import VocabularyPersistencePort


class JsonVocabularyAdapter(VocabularyPersistencePort):
    """
    JSON infrastructure adapter implementing VocabularyPersistencePort.

    Serializes and deserializes the TokenVocabulary entity to/from
    the filesystem in JSON format.
    """

    def save_vocabulary(self, vocabulary: TokenVocabulary, destination_path: str) -> None:
        """
        Serializes TokenVocabulary entity to a JSON file.

        Args:
            vocabulary (TokenVocabulary): TokenVocabulary entity to save.
            destination_path (str): File system path target.

        Raises:
            DomainError: If filesystem write or serialization fails.
        """
        if not isinstance(vocabulary, TokenVocabulary):
            raise DomainError("Input must be a TokenVocabulary instance.")

        try:
            path = Path(destination_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(vocabulary.token_to_index, f, indent=2, ensure_ascii=False)
        except (OSError, TypeError) as e:
            raise DomainError(f"Failed to save vocabulary to '{destination_path}': {str(e)}") from e

    def load_vocabulary(self, source_path: str) -> TokenVocabulary:
        """
        Deserializes JSON payload from disk and instantiates TokenVocabulary.

        Args:
            source_path (str): File system path source.

        Returns:
            TokenVocabulary: Validated domain value object.

        Raises:
            DomainError: If file is missing, invalid JSON, or violates vocabulary invariants.
        """
        try:
            path = Path(source_path)
            with path.open("r", encoding="utf-8") as f:
                raw_payload = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise DomainError(f"Failed to load vocabulary from '{source_path}': {str(e)}") from e

        if not isinstance(raw_payload, dict):
            raise DomainError(
                f"Invalid vocabulary payload in '{source_path}'. Expected dictionary."
            )


        token_to_index: dict[str, int] = raw_payload
        index_to_token: dict[int, str] = {
            int(idx): token for token, idx in token_to_index.items()
        }

        try:
            return TokenVocabulary(token_to_index=token_to_index, index_to_token=index_to_token)
        except VocabularyInvariantError as e:
            raise DomainError(f"Loaded vocabulary payload violates invariants: {str(e)}") from e
