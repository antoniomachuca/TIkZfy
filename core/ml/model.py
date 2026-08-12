"""Vision encoder and causal autoregressive TikZ decoder."""

from typing import cast

import torch
from torch import nn

from core.exceptions import TensorTopologyError, VocabularyInvariantError
from core.models import (
    BOS_INDEX,
    EOS_INDEX,
    PAD_INDEX,
    UNK_INDEX,
    ImageTensor,
    TikzTokens,
    TokenVocabulary,
)


class VisionEncoder(nn.Module):
    """Convolutional image encoder returning normalized visual tokens."""

    def __init__(self, input_channels: int, model_dimension: int) -> None:
        super().__init__()
        self.network: nn.Sequential = nn.Sequential(
            nn.Conv2d(input_channels, model_dimension, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(model_dimension, model_dimension, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.normalization: nn.LayerNorm = nn.LayerNorm(model_dimension)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into visual tokens with shape ``(B, S, D)``."""
        features: torch.Tensor = cast(torch.Tensor, self.network(images))
        batch_size, channels, height, width = features.shape
        visual_tokens: torch.Tensor = features.reshape(batch_size, channels, height * width)
        return cast(torch.Tensor, self.normalization(visual_tokens.transpose(1, 2)))


class AutoregressiveDecoder(nn.Module):
    """Causal Transformer decoder attending to the visual token sequence."""

    def __init__(
        self,
        vocabulary_size: int,
        model_dimension: int,
        max_length: int,
        num_layers: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.max_length: int = max_length
        self.token_embedding: nn.Embedding = nn.Embedding(vocabulary_size, model_dimension)
        self.position_embedding: nn.Embedding = nn.Embedding(max_length, model_dimension)
        decoder_layer: nn.TransformerDecoderLayer = nn.TransformerDecoderLayer(
            d_model=model_dimension,
            nhead=num_heads,
            dim_feedforward=model_dimension * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer: nn.TransformerDecoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )
        self.normalization: nn.LayerNorm = nn.LayerNorm(model_dimension)
        self.output_projection: nn.Linear = nn.Linear(model_dimension, vocabulary_size)

    def forward(
        self, visual_tokens: torch.Tensor, target_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Return causal token logits with shape ``(B, L, V)``."""
        sequence_length: int = target_tokens.shape[1]
        if sequence_length > self.max_length:
            raise TensorTopologyError(
                f"Target sequence length {sequence_length} exceeds {self.max_length}."
            )

        positions: torch.Tensor = torch.arange(
            sequence_length,
            device=target_tokens.device,
            dtype=torch.long,
        ).unsqueeze(0)
        token_embeddings: torch.Tensor = self.token_embedding(target_tokens)
        position_embeddings: torch.Tensor = self.position_embedding(positions)
        decoder_input: torch.Tensor = token_embeddings + position_embeddings
        causal_mask: torch.Tensor = torch.triu(
            torch.ones(
                (sequence_length, sequence_length),
                device=target_tokens.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        decoded_tokens: torch.Tensor = cast(
            torch.Tensor,
            self.transformer(
                tgt=decoder_input,
                memory=visual_tokens,
                tgt_mask=causal_mask,
            ),
        )
        normalized_tokens: torch.Tensor = cast(
            torch.Tensor, self.normalization(decoded_tokens)
        )
        return cast(torch.Tensor, self.output_projection(normalized_tokens))


class VisionAutoregressiveModel(nn.Module):
    """Small image-to-TikZ Transformer model.

    Image shape: ``(B, C, H, W)``.
    Target shape: ``(B, L)``.
    Logit shape: ``(B, L, V)``.
    """

    def __init__(
        self,
        vocabulary: TokenVocabulary,
        input_channels: int = 3,
        model_dimension: int = 128,
        max_length: int = 512,
        num_layers: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if not isinstance(vocabulary, TokenVocabulary):
            raise TypeError("Vocabulary must be a TokenVocabulary instance.")
        if input_channels <= 0 or model_dimension <= 0 or num_layers <= 0:
            raise VocabularyInvariantError(
                "input_channels, model_dimension, and num_layers must be positive."
            )
        if max_length < 2:
            raise VocabularyInvariantError("max_length must be at least 2.")
        if num_heads <= 0 or model_dimension % num_heads != 0:
            raise VocabularyInvariantError(
                "model_dimension must be divisible by a positive num_heads."
            )

        self.vocabulary: TokenVocabulary = vocabulary
        self.max_length: int = max_length
        self.encoder: VisionEncoder = VisionEncoder(input_channels, model_dimension)
        self.decoder: AutoregressiveDecoder = AutoregressiveDecoder(
            vocabulary_size=len(vocabulary.token_to_index),
            model_dimension=model_dimension,
            max_length=max_length,
            num_layers=num_layers,
            num_heads=num_heads,
        )

    def forward(
        self, images: torch.Tensor | ImageTensor, target_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Run the encoder and causal decoder under teacher forcing."""
        image_tensor: torch.Tensor = self._extract_images(images)
        if target_tokens.ndim != 2 or target_tokens.dtype != torch.long:
            raise TensorTopologyError("Target tokens must be a rank-2 torch.long tensor.")
        if image_tensor.shape[0] != target_tokens.shape[0]:
            raise TensorTopologyError("Image and target batch dimensions must match.")

        visual_tokens: torch.Tensor = self.encoder(image_tensor)
        return cast(torch.Tensor, self.decoder(visual_tokens, target_tokens))

    @torch.inference_mode()
    def generate_markup(self, image: ImageTensor) -> TikzTokens:
        """Generate a bounded greedy TikZ sequence for one image."""
        image_tensor: torch.Tensor = self._extract_images(image)
        if image_tensor.shape[0] != 1:
            raise TensorTopologyError("Inference requires an image batch of size one.")

        generated: torch.Tensor = torch.full(
            (1, 1), BOS_INDEX, dtype=torch.long, device=image_tensor.device
        )
        visual_tokens: torch.Tensor = self.encoder(image_tensor)
        step: int = 0
        finished: torch.Tensor = torch.zeros(1, dtype=torch.bool, device=image_tensor.device)
        while step < self.max_length - 1 and not bool(finished.all().item()):
            logits: torch.Tensor = self.decoder(visual_tokens, generated)
            next_token: torch.Tensor = logits[:, -1, :].argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, EOS_INDEX), next_token)
            generated = torch.cat((generated, next_token.unsqueeze(1)), dim=1)
            finished = finished | next_token.eq(EOS_INDEX)
            step += 1

        special_indices: set[int] = {BOS_INDEX, EOS_INDEX, PAD_INDEX, UNK_INDEX}
        decoded_tokens: list[str] = [
            self.vocabulary.index_to_token[index]
            for index in generated[0].tolist()
            if index not in special_indices and index in self.vocabulary.index_to_token
        ]
        begin_token: str = r"\begin{tikzpicture}"
        end_token: str = r"\end{tikzpicture}"
        content_tokens: list[str] = [
            token for token in decoded_tokens if token not in (begin_token, end_token)
        ]
        markup_tokens: list[str] = [begin_token, *content_tokens, end_token]
        return TikzTokens(markup=" ".join(markup_tokens))

    @staticmethod
    def _extract_images(images: torch.Tensor | ImageTensor) -> torch.Tensor:
        """Return and validate an image tensor with shape ``(B, C, H, W)``."""
        image_tensor: torch.Tensor = (
            images.raw_tensor if isinstance(images, ImageTensor) else images
        )
        if not isinstance(image_tensor, torch.Tensor) or image_tensor.ndim != 4:
            raise TensorTopologyError("Images must be a rank-4 tensor with shape (B, C, H, W).")
        return image_tensor
