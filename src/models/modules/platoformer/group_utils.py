import math

import torch
from torch import Tensor


def _get_trivial_elements() -> Tensor:
    """Returns the single element of the trivial group (the identity)."""
    return torch.eye(3, dtype=torch.float32).unsqueeze(0)


def _get_kleinba_elements() -> Tensor:
    """Returns the 4 rotation matrices of the Kleinba group."""
    return torch.tensor(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            [[1, 0, 0], [0, 0, 1], [1, 0, 0]],
        ],
        dtype=torch.float32,
    )


def _get_tetrahedral_elements() -> Tensor:
    """Returns the 12 rotation matrices of the Tetrahedral group."""
    return torch.tensor(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
            [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
            [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
            [[0, 0, -1], [1, 0, 0], [0, -1, 0]],
            [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
            [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
            [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
            [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
            [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
        ],
        dtype=torch.float32,
    )


def _get_octahedral_elements() -> Tensor:
    """Returns the 24 rotation matrices of the Octahedral group."""
    base = _get_tetrahedral_elements()
    # A representative element not in the tetrahedral subgroup
    c = torch.tensor(
        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float32
    ) @ torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=torch.float32)
    elements = torch.cat([base, torch.stack([b @ c for b in base])], dim=0)
    return elements


def _get_icosahedral_elements() -> Tensor:
    """
    Generates the 60 rotation matrices of the Icosahedral group programmatically
    using coset decomposition of its tetrahedral subgroup.
    """

    def _rodrigues_rotation(axis: torch.Tensor, angle: float) -> torch.Tensor:
        """Generates a rotation matrix using Rodrigues' rotation formula."""
        axis = axis / torch.linalg.norm(axis)
        K = torch.tensor(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
            dtype=torch.float32,
        )
        I = torch.eye(3, dtype=torch.float32)
        R = I + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
        return R

    T = _get_tetrahedral_elements()
    phi = (1 + math.sqrt(5)) / 2.0
    c = _rodrigues_rotation(torch.tensor([phi, 1.0, 0.0]), 2 * math.pi / 5)

    c_powers = [torch.eye(3, dtype=torch.float32)]
    for _ in range(4):
        c_powers.append(c_powers[-1] @ c)

    icosahedral_elements = torch.stack([t @ c_pow for t in T for c_pow in c_powers])

    # Find unique elements to handle potential floating point duplicates
    unique_elements = []
    atol = 1e-5
    for g in icosahedral_elements:
        is_new = all(
            not torch.allclose(g, existing_g, atol=atol)
            for existing_g in unique_elements
        )
        if is_new:
            unique_elements.append(g)

    if len(unique_elements) != 60:
        raise RuntimeError(
            f"Failed to generate Icosahedral group. Expected 60 elements, got {len(unique_elements)}"
        )

    return torch.stack(unique_elements)


def _get_axis_aligned_reflection_elements() -> Tensor:
    """
    Returns the 8 diagonal matrices with +/-1 on the diagonal (C2 x C2 x C2 group).
    These are reflections across coordinate planes and inversions.
    """
    elements = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                elements.append(
                    torch.tensor(
                        [[sx, 0, 0], [0, sy, 0], [0, 0, sz]], dtype=torch.float32
                    )
                )
    return torch.stack(elements)
