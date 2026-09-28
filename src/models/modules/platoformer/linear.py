import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

# Import the pre-computed group data
from .groups import PlatonicSolidGroup


class PlatonicLinear(nn.Module):
    """
    A Linear layer constrained to be a group convolution over a Platonic Solid group.
    This version includes a corrected initialization scheme to preserve variance.
    """

    __constants__ = ["has_bias"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group: PlatonicSolidGroup,
        bias: bool = True,
    ):
        if in_features % group.G != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by the group order {group.G} for solid '{group}'."
            )
        if out_features % group.G != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by the group order {group.G} for solid '{group}'."
            )

        super().__init__()

        self.group = group

        self.has_bias = bias

        self.in_features = in_features
        self.out_features = out_features

        self.in_channels = in_features // group.G
        self.out_channels = out_features // group.G

        self.linear = nn.Linear(self.in_channels, self.out_channels, bias=bias)

        self.kernel = nn.Parameter(
            torch.empty(group.G, self.out_channels, self.in_channels)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels))
        else:
            self.register_parameter("bias", None)

        self._init_kernel()

    def _init_kernel(self) -> None:
        """
        Initialize the kernel and bias with variance-preserving scaling.

        Standard initializers (like Kaiming) fail to correctly infer the
        effective fan-in of the full weight matrix. We must calculate it
        manually as (group_size * in_channels_per_group).
        """
        # Calculate the effective fan-in for the full weight matrix.
        fan_in = self.group.G * self.in_channels

        # Initialize the kernel from a normal distribution. The std is calculated
        # to ensure the output variance is approximately equal to the input variance.
        std = 1.0 / math.sqrt(fan_in)
        nn.init.normal_(self.kernel, mean=0.0, std=std)

        if self.bias is not None:
            # Initialize bias using the same correct fan-in.
            if fan_in > 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def get_weight(self) -> Tensor:
        """
        Constructs the full [G*O, G*I] weight matrix from the fundamental kernel.
        """
        inv_g_indices = self.group.inverse_indices[self.group.indices[None]]
        kernel_group_idx = self.group.cayley_table[
            self.group.indices[:, None], inv_g_indices
        ]

        expanded_kernel = self.kernel[kernel_group_idx]

        weight = expanded_kernel.permute(0, 2, 1, 3).reshape(
            self.out_features, self.in_features
        )

        return weight

    def forward(self, x: Tensor) -> Tensor:
        """Applies the group-equivariant linear transformation."""
        weight = self.get_weight()
        output = F.linear(x, weight, None)

        if self.has_bias:
            output_shape = output.shape
            output = output.view(*output_shape[:-1], self.group.G, self.out_channels)
            output = output + self.bias
            output = output.view(output_shape)

        return output

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(G={self.group.G}, in_features={self.in_features}, out_features={self.out_features}, bias={self.has_bias})"
