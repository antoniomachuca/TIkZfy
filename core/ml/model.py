"""Vision encoder and causal autoregressive TikZ decoder."""

from typing import cast

import torch
import torch.nn.functional as F
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


class ConvResidualBlock(nn.Module):
    """Residual convolutional block with LayerNorm and GELU.

    Structure: ``Conv2D -> LayerNorm -> GELU -> Residual``.
    Tensor Shape: ``(B, C, H, W) -> (B, C, H, W)``.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise VocabularyInvariantError("channels must be positive.")
        self.conv: nn.Conv2d = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.norm: nn.LayerNorm = nn.LayerNorm(channels)
        self.activation: nn.GELU = nn.GELU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply residual block to input tensor with shape ``(B, C, H, W)``."""
        # Shape: (B, C, H, W)
        residual: torch.Tensor = features
        conv_out: torch.Tensor = cast(torch.Tensor, self.conv(features))
        # Channels-last permutation for spatial LayerNorm: (B, C, H, W) -> (B, H, W, C)
        norm_in: torch.Tensor = conv_out.permute(0, 2, 3, 1)
        norm_out: torch.Tensor = cast(torch.Tensor, self.norm(norm_in))
        # Permute back to standard PyTorch spatial layout: (B, H, W, C) -> (B, C, H, W)
        norm_spatial: torch.Tensor = norm_out.permute(0, 3, 1, 2)
        activated: torch.Tensor = cast(torch.Tensor, self.activation(norm_spatial))
        return residual + activated


def build_2d_sinusoidal_positional_encoding(
    height: int,
    width: int,
    dimension: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute 2D sinusoidal (sine/cosine) positional encodings for a 2D spatial grid.

    Generates separable spatial frequency bases for Y and X Cartesian axes in logical O(1):
        PE_Y(y, 2i)   = sin(y / 10000^(4i/D)), PE_Y(y, 2i+1) = cos(y / 10000^(4i/D))
        PE_X(x, 2i)   = sin(x / 10000^(4i/D)), PE_X(x, 2i+1) = cos(x / 10000^(4i/D))
        PE_2D(y, x, :) = [PE_Y(y, :), PE_X(x, :)]

    Args:
        height (int): Grid spatial height H'.
        width (int): Grid spatial width W'.
        dimension (int): Feature channel dimension D.
        device (torch.device): Target execution device.
        dtype (torch.dtype): Floating-point tensor precision.

    Returns:
        torch.Tensor: Positional encoding tensor with shape ``(1, H' * W', D)``.

    Spatial complexity: O(H' * W' * D).
    Temporal complexity: O(1) logical GPU parallel algebra.
    """
    if dimension % 4 != 0:
        raise TensorTopologyError(
            "dimension must be divisible by 4 for balanced 2D sinusoidal encodings."
        )

    dim_y: int = dimension // 2
    dim_x: int = dimension - dim_y

    freq_indices_y: torch.Tensor = torch.arange(0, dim_y, 2, device=device, dtype=dtype)
    omega_y: torch.Tensor = 1.0 / (10000.0 ** (freq_indices_y / dim_y))

    freq_indices_x: torch.Tensor = torch.arange(0, dim_x, 2, device=device, dtype=dtype)
    omega_x: torch.Tensor = 1.0 / (10000.0 ** (freq_indices_x / dim_x))

    pos_y: torch.Tensor = torch.arange(height, device=device, dtype=dtype)
    pos_x: torch.Tensor = torch.arange(width, device=device, dtype=dtype)

    out_y: torch.Tensor = torch.einsum("h,d->hd", pos_y, omega_y)
    out_x: torch.Tensor = torch.einsum("w,d->wd", pos_x, omega_x)

    pe_y: torch.Tensor = torch.stack((torch.sin(out_y), torch.cos(out_y)), dim=-1).reshape(
        height, dim_y
    )
    pe_x: torch.Tensor = torch.stack((torch.sin(out_x), torch.cos(out_x)), dim=-1).reshape(
        width, dim_x
    )

    pe_y_grid: torch.Tensor = pe_y.unsqueeze(1).expand(height, width, dim_y)
    pe_x_grid: torch.Tensor = pe_x.unsqueeze(0).expand(height, width, dim_x)

    pe_2d: torch.Tensor = torch.cat((pe_y_grid, pe_x_grid), dim=-1)
    return pe_2d.reshape(1, height * width, dimension)


class VisionEncoder(nn.Module):
    """Deep convolutional image encoder with CoordConv and 2D Positional Encoding.

    Downsamples the input image through a 2-stage convolutional stem conditioned on
    spatial coordinate planes (X, Y in [-1, 1]) and processes the feature maps
    through a sequence of residual blocks.

    Input shape: ``(B, C_{in}, H, W)``
    Output shape: ``(B, S, D)`` where ``S = (H / 4) * (W / 4)`` and ``D = model_dimension``.
    """

    def __init__(
        self,
        input_channels: int,
        model_dimension: int,
        num_blocks: int = 6,
        use_coord_conv: bool = True,
        use_2d_pos_encoding: bool = True,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or model_dimension <= 0:
            raise VocabularyInvariantError("input_channels and model_dimension must be positive.")
        if num_blocks < 0:
            raise VocabularyInvariantError("num_blocks must be non-negative.")

        self.use_coord_conv: bool = use_coord_conv
        self.use_2d_pos_encoding: bool = use_2d_pos_encoding
        stem_in_channels: int = input_channels + 2 if use_coord_conv else input_channels

        self.stem: nn.Sequential = nn.Sequential(
            nn.Conv2d(stem_in_channels, model_dimension, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(model_dimension, model_dimension, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.residual_blocks: nn.Sequential = nn.Sequential(
            *[ConvResidualBlock(channels=model_dimension) for _ in range(num_blocks)]
        )
        self.normalization: nn.LayerNorm = nn.LayerNorm(model_dimension)

    @staticmethod
    def _add_coordinate_channels(images: torch.Tensor) -> torch.Tensor:
        """Inject normalized [-1, 1] Cartesian coordinate planes (X, Y) into channels."""
        batch_size, _, height, width = images.shape
        y_coords: torch.Tensor = torch.linspace(
            -1.0, 1.0, steps=height, device=images.device, dtype=images.dtype
        )
        x_coords: torch.Tensor = torch.linspace(
            -1.0, 1.0, steps=width, device=images.device, dtype=images.dtype
        )
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        coord_x: torch.Tensor = (
            grid_x.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, height, width)
        )
        coord_y: torch.Tensor = (
            grid_y.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, height, width)
        )
        return torch.cat((images, coord_x, coord_y), dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into visual tokens with shape ``(B, S, D)``."""
        x: torch.Tensor = self._add_coordinate_channels(images) if self.use_coord_conv else images
        # Shape: (B, C_in + 2, H, W) -> (B, D, H/4, W/4)
        features: torch.Tensor = cast(torch.Tensor, self.stem(x))
        # Shape: (B, D, H/4, W/4) -> (B, D, H/4, W/4)
        features = cast(torch.Tensor, self.residual_blocks(features))
        batch_size, channels, height, width = features.shape
        # Spatial flatten to token sequence: (B, D, H/4, W/4) -> (B, D, S) -> (B, S, D)
        visual_tokens: torch.Tensor = features.reshape(
            batch_size, channels, height * width
        ).transpose(1, 2)
        if self.use_2d_pos_encoding:
            pos_encoding: torch.Tensor = build_2d_sinusoidal_positional_encoding(
                height=height,
                width=width,
                dimension=channels,
                device=features.device,
                dtype=features.dtype,
            )
            visual_tokens = visual_tokens + pos_encoding
        return cast(torch.Tensor, self.normalization(visual_tokens))


def resolve_device(device: torch.device | str | None = None) -> torch.device:
    """Return the execution device, resolving to CUDA if available when unspecified."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AutoregressiveDecoder(nn.Module):
    """Causal Transformer decoder attending to the visual token sequence.

    Supports configurable depth (6 to 8 layers), latent dimension (d_model=384),
    multi-head attention (n_head=8), and feed-forward dimension (d_ff=1536).
    """

    def __init__(
        self,
        vocabulary_size: int,
        model_dimension: int,
        max_length: int,
        num_layers: int,
        num_heads: int,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if vocabulary_size <= 0 or model_dimension <= 0 or num_layers <= 0:
            raise VocabularyInvariantError(
                "vocabulary_size, model_dimension, and num_layers must be positive."
            )
        if max_length < 2:
            raise VocabularyInvariantError("max_length must be at least 2.")
        if num_heads <= 0 or model_dimension % num_heads != 0:
            raise VocabularyInvariantError(
                "model_dimension must be divisible by a positive num_heads."
            )
        ff_dimension: int = dim_feedforward if dim_feedforward is not None else model_dimension * 4
        if ff_dimension <= 0:
            raise VocabularyInvariantError("dim_feedforward must be positive.")
        if not 0.0 <= dropout <= 1.0:
            raise VocabularyInvariantError("dropout must be in range [0.0, 1.0].")

        self.max_length: int = max_length
        self.model_dimension: int = model_dimension
        self.num_layers: int = num_layers
        self.num_heads: int = num_heads
        self.dim_feedforward: int = ff_dimension
        self.dropout_p: float = dropout
        self.token_embedding: nn.Embedding = nn.Embedding(vocabulary_size, model_dimension)
        self.position_embedding: nn.Embedding = nn.Embedding(max_length, model_dimension)
        decoder_layer: nn.TransformerDecoderLayer = nn.TransformerDecoderLayer(
            d_model=model_dimension,
            nhead=num_heads,
            dim_feedforward=ff_dimension,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer: nn.TransformerDecoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )
        self.normalization: nn.LayerNorm = nn.LayerNorm(model_dimension)
        self.output_projection: nn.Linear = nn.Linear(model_dimension, vocabulary_size)

    def forward(self, visual_tokens: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
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
        # Embedding dropout: prevents memorization of autoregressive token sequences
        decoder_input = F.dropout(decoder_input, p=self.dropout_p, training=self.training)
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
        normalized_tokens: torch.Tensor = cast(torch.Tensor, self.normalization(decoded_tokens))
        return cast(torch.Tensor, self.output_projection(normalized_tokens))


class VisionAutoregressiveModel(nn.Module):
    """Multimodal image-to-TikZ Transformer model.

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
        dim_feedforward: int | None = None,
        num_encoder_blocks: int = 6,
        use_coord_conv: bool = True,
        use_2d_pos_encoding: bool = True,
        dropout: float = 0.1,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(vocabulary, TokenVocabulary):
            raise TypeError("Vocabulary must be a TokenVocabulary instance.")
        if input_channels <= 0 or model_dimension <= 0 or num_layers <= 0:
            raise VocabularyInvariantError(
                "input_channels, model_dimension, and num_layers must be positive."
            )
        if num_encoder_blocks < 0:
            raise VocabularyInvariantError("num_encoder_blocks must be non-negative.")
        if max_length < 2:
            raise VocabularyInvariantError("max_length must be at least 2.")
        if num_heads <= 0 or model_dimension % num_heads != 0:
            raise VocabularyInvariantError(
                "model_dimension must be divisible by a positive num_heads."
            )

        self.vocabulary: TokenVocabulary = vocabulary
        self.max_length: int = max_length
        self.model_dimension: int = model_dimension
        self.num_layers: int = num_layers
        self.num_heads: int = num_heads
        self.num_encoder_blocks: int = num_encoder_blocks
        self.use_coord_conv: bool = use_coord_conv
        self.use_2d_pos_encoding: bool = use_2d_pos_encoding
        self.target_device: torch.device = resolve_device(device)

        self.encoder: VisionEncoder = VisionEncoder(
            input_channels=input_channels,
            model_dimension=model_dimension,
            num_blocks=num_encoder_blocks,
            use_coord_conv=use_coord_conv,
            use_2d_pos_encoding=use_2d_pos_encoding,
        )
        self.decoder: AutoregressiveDecoder = AutoregressiveDecoder(
            vocabulary_size=len(vocabulary.token_to_index),
            model_dimension=model_dimension,
            max_length=max_length,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        if device is not None:
            self.to(self.target_device)

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
    def generate_markup(
        self, image: ImageTensor, device: torch.device | str | None = None
    ) -> TikzTokens:
        """Generate a bounded greedy TikZ sequence for one image."""
        image_tensor: torch.Tensor = self._extract_images(image)
        if image_tensor.shape[0] != 1:
            raise TensorTopologyError("Inference requires an image batch of size one.")

        target_device: torch.device = (
            resolve_device(device) if device is not None else image_tensor.device
        )
        image_tensor = image_tensor.to(target_device)
        generated: torch.Tensor = torch.full(
            (1, 1), BOS_INDEX, dtype=torch.long, device=target_device
        )
        visual_tokens: torch.Tensor = self.encoder(image_tensor)
        step: int = 0
        finished: torch.Tensor = torch.zeros(1, dtype=torch.bool, device=target_device)
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
