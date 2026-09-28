import torch
from torch import Tensor

from .groups import PlatonicSolidGroup


def lift_scalars(x: Tensor, group: PlatonicSolidGroup):
    return x.unsqueeze(2).repeat(
        1, 1, group.G, 1
    )  # (B, N, C_hidden_g) -> (B, N, G, C_hidden_g)


def lift_vectors(x: Tensor, group: PlatonicSolidGroup):
    frames = group.elements  # (G, 3, 3)

    return torch.einsum("gij,...cj->...gci", frames, x).flatten(
        -2, -1
    )  # (..., C, 3) -> (..., G, C, 3) -> (..., G , C * 3)


def readout_scalars(x: Tensor, _):
    return x.mean(dim=-2)  # (..., G * C) -> (..., G, C) -> (..., C)


def readout_vectors(x: Tensor, group: PlatonicSolidGroup):
    x = x.unflatten(-1, (-1, 3))  # (..., G * C) -> (..., G, C, 3)

    frames = group.elements  # (G, 3, 3)

    return (
        torch.einsum("gji,...gcj->...ci", frames, x) / group.G
    )  # frame transposed, result: (..., c, 3)


def readout(x, num_scalars, num_vectors, group):
    x = x.unflatten(-1, (group.G, -1))

    x_scalars, x_vectors = x.split([num_scalars, num_vectors * 3], dim=-1)
    # (..., G * C) -> (..., G, C)

    scalars = readout_scalars(x_scalars, group)
    vectors = readout_vectors(x_vectors, group)

    return scalars, vectors  # (..., C), (..., C, 3)
