import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

# Assumes these modules are in your project structure
from .rope import PlatonicRoPE
from .linear import PlatonicLinear
from .groups import PlatonicSolidGroup


class PlatonicConv(nn.Module):
    """
    Computes a group-equivariant dynamic convolution supporting both graph and dense modes.

    This layer uses Rotary Positional Embeddings (RoPE) to compute a dynamic
    convolution kernel for each element in a batch (e.g., each point cloud or
    padded sequence). It is equivariant to the symmetries of a specified Platonic solid.

    The layer operates on "group feature maps" with flattened dimensions [..., G*C],
    and handles both graph-structured data (via a `batch` tensor) and dense,
    padded data (via a `mask` tensor).
    """

    __constants__ = ["context_channels"]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        num_heads: int,
        group: PlatonicSolidGroup,
        context_channels: (
            int | None
        ) = None,  # if set, will perform cross attend between input queries and context
        spatial_dims: int = 3,
        freq_sigma: float = 1.0,
        learned_freqs: bool = True,
        bias: bool = True,
        mean_aggregation: bool = False,
    ):
        # --- Dimension Validation and Setup ---
        if in_channels % group.G != 0:
            raise ValueError(
                f"in_channels ({in_channels}) must be divisible by group size ({group.G})."
            )
        if out_channels % group.G != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by group size ({group.G})."
            )
        if embed_dim % (group.G * num_heads * 2) != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by (group_size * num_heads * 2) = "
                f"{group.G * num_heads * 2} for RoPE."
            )

        super().__init__()
        # --- Group Setup ---

        self.group = group
        self.context_channels = context_channels

        self.in_channels_g = in_channels // self.group.G
        self.out_channels_g = out_channels // self.group.G
        self.context_channels_g = (
            context_channels // self.group.G if context_channels else None
        )

        self.embed_dim = embed_dim
        self.embed_dim_g = embed_dim // self.group.G
        self.head_dim = self.embed_dim_g // num_heads

        self.out_channels = out_channels
        self.num_heads = num_heads

        self.mean_aggregation = mean_aggregation

        # --- Sub-modules ---
        self.q_proj = PlatonicLinear(in_channels, embed_dim, group, bias=bias)
        self.v_proj = PlatonicLinear(
            in_channels if context_channels is None else context_channels,
            embed_dim,
            group,
            bias=bias,
        )

        # Group-equivariant RoPE for positional information
        self.rope_emb = PlatonicRoPE(
            embed_dim=embed_dim,
            num_heads=num_heads,
            group=group,
            spatial_dims=spatial_dims,
            freq_sigma=freq_sigma,
            learned_freqs=learned_freqs,
        )

        # Final equivariant linear layer (acts on the full G*dim space)
        self.out_proj = PlatonicLinear(embed_dim, out_channels, group, bias=bias)

    def _compute_qkv(
        self, x: Tensor, pos: Tensor, x_context: Tensor, pos_context: Tensor
    ):
        """Shared logic for projections and RoPE application."""
        # Get leading dimensions (e.g., [N] for graph, [B, S] for dense)
        leading_dims = x.shape[:-1]
        leading_dims_context = x_context.shape[:-1]

        # Query and value projections
        q_raw = self.q_proj(x)  # [..., G * H * D_h]
        v_raw = self.v_proj(x_context)  # [..., G * H * D_h]

        # Reshape to merge G and H axes for RoPE module: [..., G, E_g] -> [..., G*H, D_h]
        q = q_raw.reshape(*leading_dims, self.group.G, self.num_heads, self.head_dim)
        v = v_raw.reshape(
            *leading_dims_context, self.group.G, self.num_heads, self.head_dim
        )
        k = torch.ones_like(v)

        # Apply RoPE to query and key
        q_rope = self.rope_emb(q, pos)
        k_rope = self.rope_emb(k, pos_context)

        return q_rope, k_rope, v

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        x_context: Tensor | None,
        pos_context: Tensor | None,
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        Applies dynamic group convolution, dispatching to graph or dense mode.

        Args:
            x (Tensor): Input features. (N, G*I) for graph, (B, S, G*I) for dense.
            pos (Tensor): Positions. (N, D_spatial) or (B, S, D_spatial).
            batch (Optional[Tensor]): For graph mode. Batch index for each point (N,).
            mask (Optional[Tensor]): For dense mode. Boolean mask (B, S).

        Returns:
            Tensor: Output features, (N, G*O) or (B, S, G*O).
        """
        if self.context_channels is None:
            x_context, pos_context = x, pos

        # B: batch size, S: sequence length
        q_rope, k_rope, v = self._compute_qkv(x, pos, x_context, pos_context)
        v = v if mask is None else v * mask[..., None, None, None]

        # Aggregate across sequence length to get one kernel per sample
        kv_kernel = torch.einsum("bsghd,bsghe->bghde", k_rope, v)

        # Normalize kernel by number of nodes
        kv_kernel = (
            kv_kernel / x_context.shape[1]
        )  # TODO should be x_context.shape[1], fuck me

        # Apply kernel
        output = torch.einsum("bsghd,bghde->bsghe", q_rope, kv_kernel)
        output = output.flatten(-3, -1)  # -> (..., G, H, H_dim) -> (..., G*H*H_dim)

        # Final equivariant projection
        return self.out_proj(output)
