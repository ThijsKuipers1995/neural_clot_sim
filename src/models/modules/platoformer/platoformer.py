import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from .block import PlatonicBlock

from .groups import PlatonicSolidGroup
from .linear import PlatonicLinear
from .lift_and_readout import (
    lift_scalars,
    lift_vectors,
    readout,
)


class PlatonicTransformer(nn.Module):
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

    __constants__ = ["input_dim_vec", "output_dim", "output_dim_vec", "task_level"]

    def __init__(
        self,
        input_dim: int,
        input_dim_vec: int,
        hidden_dim: int,
        output_dim: int,
        output_dim_vec: int,
        nhead: int,
        num_layers: int,
        solid_name: str,
        ffn_dim_factor: int = 4,
        task_level: str = "node",
        dropout: float = 0.1,
        norm_first: bool = True,
        freq_sigma: float = 1.0,
        learned_freqs: bool = True,
        spatial_dim: int = 3,
        **kwargs
    ):
        if task_level not in ["node", "graph"]:
            raise ValueError("task_level must be 'node' or 'graph'.")

        super().__init__()

        # --- Group and Dimension Setup ---
        self.group = PlatonicSolidGroup(solid_name.lower())

        self.input_dim_vec = input_dim_vec

        self.hidden_dim = hidden_dim
        self.hidden_dim_per_g = (
            hidden_dim // self.group.G
        )  # Per-group-element channel dimension

        self.task_level = task_level
        self.output_dim = output_dim
        self.output_dim_vec = output_dim_vec

        # --- Modules ---
        # 1. Input Embedding: Applied before lifting to the group.
        # Maps input features to the per-group-element hidden dimension.
        self.x_embedder = PlatonicLinear(
            (input_dim + input_dim_vec * spatial_dim) * self.group.G,
            self.hidden_dim,
            self.group,
            bias=False,
        )

        # 2. Equivariant Encoder Layers
        # The blocks operate on the total flattened dimension (G * C).
        dim_feedforward = self.hidden_dim * ffn_dim_factor

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
                    **kwargs
                )
            )

        # 3. Readout Head
        self.readout = PlatonicLinear(
            self.hidden_dim,
            self.group.G * (output_dim + output_dim_vec * spatial_dim),
            self.group,
        )

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        vec: Optional[Tensor] = None,
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

        # 1. Convert to dense format if needed

        # 2. Lift scalars and vectors, then embed
        x = lift_scalars(x, self.group)

        if self.input_dim_vec != 0:
            vec = lift_vectors(x, vec, self.group)
            x = torch.cat((x, vec), dim=-1)

        x = x.flatten(-2, -1)

        x = self.x_embedder(x)

        # 3. Equivariant Encoder (Platonic Conv Blocks)
        for layer in self.layers:
            x = layer(x=x, pos=pos)

        # 4. Readout layer
        x = self.readout(x)

        # 5. Pooling of the nodes if needed
        # Normalize by number of nodes in each graph (or not, but then rescale by avg_num_nodes)
        x = x.mean(dim=1) if self.task_level == "graph" else x

        # 6. Extract the scalar and vector parts
        scalars, vectors = readout(x, self.output_dim, self.output_dim_vec, self.group)

        return scalars, vectors
