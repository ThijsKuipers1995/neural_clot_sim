import torch
import torch.nn as nn

from torch import Tensor

from torch_cluster import fps

from .block import PlatonicBlock

from .groups import PlatonicSolidGroup
from .linear import PlatonicLinear
from .lift_and_readout import (
    lift_scalars,
    lift_vectors,
    readout,
)


class PlatonicPerciever(nn.Module):
    """
    A Transformer architecture equivariant to the symmetries of a specified Platonic solid.

    This model processes point cloud data. It first embeds input node features, then
    "lifts" them into a group-equivariant feature space. A series of PlatonicBlocks
    process these features equivariantly. Finally, for graph-level tasks, it pools
    over the nodes and the group to produce a single invariant prediction. For node-level
    tasks, it pools over the group axis to produce invariant node predictions.

    Args:
        input_dim (int): Dimensionality of the initial node features.
        hidden_dim (int): The per-group-element channel dimension used throughout the model.
        num_latents (int): number of latent points in latent pointcloud
        output_dim (int): Dimensionality of the final output.
        nhead (int): Number of attention heads in each PlatonicBlock.
        num_layers (int): Number of PlatonicBlock layers.
        solid_name (str): The name of the Platonic solid ('tetrahedron', 'octahedron',
                          'icosahedron') to define the symmetry group.
        ffn_dim_factor (int): Multiplier for the feed-forward network's hidden dimension,
                              relative to `hidden_dim`.
        task_level (str): "node" or "graph". Determines the pooling strategy.
        dropout (float): Dropout rate.
        norm_first (bool): If True, use pre-normalization in the blocks.
        **kwargs: Additional keyword arguments for the PlatonicBlock layers
    """

    __constants__ = ["input_dim_vec", "output_dim", "output_dim_vec", "num_latents"]

    def __init__(
        self,
        input_dim: int,
        input_dim_vec: int,
        num_latents: int,
        hidden_dim: int,
        output_dim: int,
        output_dim_vec: int,
        nhead: int,
        num_layers: int,
        solid_name: str,
        num_encoder_layers: int = 0,
        ffn_dim_factor: int = 4,
        dropout: float = 0.1,
        norm_first: bool = True,
        freq_sigma: float = 1.0,
        learned_freqs: bool = True,
        spatial_dim: int = 3,
        **kwargs,
    ):
        super().__init__()

        # --- Group and Dimension Setup ---
        self.group = PlatonicSolidGroup(solid_name.lower())

        self.input_dim_vec = input_dim_vec
        self.num_latents = num_latents

        self.hidden_dim = hidden_dim
        self.hidden_dim_per_g = (
            hidden_dim // self.group.G
        )  # Per-group-element channel dimension

        self.output_dim = output_dim
        self.output_dim_vec = output_dim_vec

        # --- Modules ---
        # 1. Input Embedding: Applied before lifting to the group.
        # Maps input features to the per-group-element hidden dimension.
        self.x_embedder = nn.Sequential(
            PlatonicLinear(
                (input_dim + input_dim_vec * spatial_dim) * self.group.G,
                self.hidden_dim,
                self.group,
                bias=False,
            ),
            nn.SiLU(),
            PlatonicLinear(
                self.hidden_dim,
                self.hidden_dim,
                self.group,
                bias=False,
            ),
        )

        dim_feedforward = self.hidden_dim * ffn_dim_factor

        # 2. Encoder Layers: full linear attention on encoding state
        self.encoder_layers = nn.ModuleList()
        for _ in range(num_encoder_layers):
            self.encoder_layers.append(
                PlatonicBlock(
                    d_model=self.hidden_dim,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    group=self.group,
                    dropout=dropout,
                    norm_first=norm_first,
                    freq_sigma=freq_sigma,
                    learned_freqs=learned_freqs,
                    spatial_dims=spatial_dim,
                    **kwargs,
                )
            )

        # 2.5. Downsample input point cloud to size num_latents
        self.downsample_block = PlatonicBlock(
            d_model=self.hidden_dim,
            d_context=self.hidden_dim,  # this enabled cross attn
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            group=self.group,
            dropout=dropout,
            norm_first=norm_first,
            freq_sigma=freq_sigma,
            learned_freqs=learned_freqs,
            spatial_dims=spatial_dim,
            **kwargs,
        )

        # 3. Equivariant Encoder Layers
        # The blocks operate on the total flattened dimension (G * C).
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                PlatonicBlock(
                    d_model=self.hidden_dim,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    group=self.group,
                    dropout=dropout,
                    norm_first=norm_first,
                    freq_sigma=freq_sigma,
                    learned_freqs=learned_freqs,
                    spatial_dims=spatial_dim,
                    **kwargs,
                )
            )

        # 4. query function
        self.query = PlatonicBlock(
            d_model=self.hidden_dim,
            d_context=self.hidden_dim,  # this enabled cross attn
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            group=self.group,
            dropout=dropout,
            norm_first=norm_first,
            freq_sigma=freq_sigma,
            learned_freqs=learned_freqs,
            spatial_dims=spatial_dim,
            **kwargs,
        )

        # 5. Readout Head
        self.readout = PlatonicLinear(
            self.hidden_dim,
            self.group.G * (output_dim + output_dim_vec * spatial_dim),
            self.group,
        )

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        query_x: Tensor,
        query_pos: Tensor,
        vec: Tensor | None = None,
        query_vec: Tensor | None = None,
        mask: Tensor | None = None,
        downsample_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass for the Platonic Transformer.

        Args:
            x (Tensor): Input node features of shape (N, input_dim).
            pos (Tensor): Node positions of shape (N, spatial_dims).
            batch (Tensor): Batch index for each node of shape (N,).
            mask (Tensor, optional): Attention mask of shape (B, N) or (N, N) for dense inputs.

        Returns:
            Tensor: Final predictions. Shape is (B, output_dim) for graph tasks
                    or (N, output_dim) for node tasks.
        """

        # 1. Lift and embed inputs and queries
        x, query_x = lift_scalars(x, self.group), lift_scalars(query_x, self.group)

        if self.input_dim_vec != 0:
            vec, query_vec = lift_vectors(vec, self.group), lift_vectors(
                query_vec, self.group
            )
            x = torch.cat((x, vec), dim=-1)
            query_x = torch.cat((query_x, query_vec), dim=-1)

        x, query_x = x.flatten(-2, -1), query_x.flatten(-2, -1)

        x = self.x_embedder(x)
        query_x = self.x_embedder(query_x)

        # 2. Encode x
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x=x, pos=pos, mask=mask)

        # 2.5. Downsample inputs
        B, N, D = x.shape

        if self.num_latents < N:

            batch = torch.arange(B, device=x.device).repeat_interleave(N)

            if downsample_mask is None:
                x = x.reshape(-1, D)
                pos = pos.reshape(-1, 3)

                indices = fps(pos, batch, ratio=self.num_latents / N)

                x_down = x[indices].view(B, self.num_latents, D)
                pos_down = pos[indices].view(B, self.num_latents, 3)
                if mask is not None:
                    mask_down = mask.view(-1)[indices].view(B, -1)
                else:
                    mask_down = None

                x = x.view(B, N, D)
                pos = pos.view(B, N, 3)
            else:
                downsample_mask = downsample_mask.view(-1)
                _x = x.reshape(-1, D)[downsample_mask]
                _pos = pos.reshape(-1, 3)[downsample_mask]

                _N = _x.shape[0] // B

                indices = fps(_pos, batch[downsample_mask], ratio=self.num_latents / _N)

                x_down = _x[indices].view(B, self.num_latents, D)
                pos_down = _pos[indices].view(B, self.num_latents, 3)
                mask_down = None
        else:
            x_down = x
            pos_down = pos
            mask_down = mask

        # 3. Get latent pointcloud -> B, num_latents, C * G
        latents = self.downsample_block(x_down, pos_down, x, pos, mask=mask)

        # 4. Equivariant processor (Platonic Conv Blocks)
        x = latents

        for layer in self.layers:
            x = layer(x=x, pos=pos_down, mask=mask_down)

        # 5. Apply the query function
        x = self.query(query_x, query_pos, x, pos_down, mask=mask_down)

        # 6. Readout layer
        x = self.readout(x)

        # 7. Extract the scalar and vector parts
        scalars, vectors = readout(x, self.output_dim, self.output_dim_vec, self.group)

        return scalars, vectors
