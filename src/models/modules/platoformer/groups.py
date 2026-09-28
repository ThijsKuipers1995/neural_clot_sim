import torch
from torch import Tensor
from typing import Callable, Dict
from .group_utils import (
    _get_icosahedral_elements,
    _get_axis_aligned_reflection_elements,
    _get_kleinba_elements,
    _get_octahedral_elements,
    _get_tetrahedral_elements,
    _get_trivial_elements,
)


class PlatonicSolidGroup(torch.nn.Module):
    """
    A class to hold and pre-compute the necessary data for a Platonic solid
    symmetry group, including its elements, order, inverse indices, and
    Cayley table for group multiplication.
    """

    __constants__ = ["solid_name", "G"]

    PLATONIC_GROUPS: Dict[str, Callable[[], Tensor]] = {
        "trivial": _get_trivial_elements(),
        "kleinba": _get_kleinba_elements(),
        "tetrahedron": _get_tetrahedral_elements(),
        "octahedron": _get_octahedral_elements(),
        "icosahedron": _get_icosahedral_elements(),
        "octahedron_reflections": _get_axis_aligned_reflection_elements(),  # Added the new group
    }

    def __init__(self, solid_name: str = "trivial"):
        """
        Initializes the group with a tensor of its elements.

        Args:
            group_elements (Tensor): A tensor of shape [G, 3, 3] where G is
                                     the order of the group. Each 3x3 matrix
                                     is a group element.
            solid_name (str): The name of the group.
        """
        if solid_name not in self.PLATONIC_GROUPS:
            raise ValueError(
                f"Unsupported group `{solid_name}`: supported groups are {f','.join(self.PLATONIC_GROUPS.keys())}.\n"
                "Register custom group with `PlatonicSolidGroup.register_group(solid_name: str, group_elements: Tensor)`."
            )

        super().__init__()

        self.solid_name = solid_name
        self.atol: float = 1.0e-5

        self.register_buffer("_elements", self.PLATONIC_GROUPS[solid_name])
        self._G: int = self._elements.shape[0]

        self.register_buffer(
            "_indices", torch.arange(self._elements.shape[0], dtype=torch.long)
        )
        self.register_buffer("_inverse_indices", self._compute_inverse_indices().long())
        self.register_buffer("_cayley_table", self._compute_cayley_table().long())

    @property
    def G(self) -> int:
        return self._G

    @property
    def elements(self) -> Tensor:
        return self._elements

    @property
    def indices(self) -> Tensor:
        return self._indices

    @property
    def inverse_indices(self) -> Tensor:
        return self._inverse_indices

    @property
    def cayley_table(self) -> Tensor:
        return self._cayley_table

    def register_group(
        self, solid_name: str, group_elements: Tensor, replace: bool = False
    ) -> None:
        if solid_name in self.PLATONIC_GROUPS and not replace:
            raise ValueError(f"`{solid_name}` is already registered.")

        self._check_group_integrity(group_elements)

        self.PLATONIC_GROUPS[solid_name] = group_elements

    @staticmethod
    def _check_group_integrity(self, elements: Tensor) -> None:
        try:
            dets = torch.linalg.det(elements)
            # For O(3) matrices, determinants must be +1 or -1
            if not torch.allclose(torch.abs(dets), torch.ones_like(dets), atol=1e-5):
                raise ValueError(
                    f"All elements must be orthogonal (determinant=+/-1). "
                    f"Found elements with incorrect determinants."
                )

            # Check for orthogonality: R^T R = I
            for i in range(self._G):
                if not torch.allclose(
                    elements[i].T @ elements[i],
                    torch.eye(3, dtype=torch.float64),
                    atol=1e-5,
                ):
                    raise ValueError(f"Element {i} is not an orthogonal matrix.")

        except torch.linalg.LinAlgError as e:
            raise ValueError(
                f"Could not compute properties. Elements must be a tensor containing 3x3 matrices. Error: {e}"
            )

    @torch.no_grad()
    def _compute_inverse_indices(self) -> Tensor:
        """Computes the index of the inverse for each group element."""
        _inverse_indices = torch.zeros(self._G, dtype=torch.long)

        # For orthogonal matrices, inverse is the transpose
        inverses_mat = self._elements.mT

        for i in range(self._G):
            # Find which element in the group matches the inverse
            diffs = torch.sum(
                (self._elements - inverses_mat[i].unsqueeze(0)) ** 2, dim=(1, 2)
            )
            j = torch.argmin(diffs)
            if diffs[j] >= self.atol**2:
                raise RuntimeError(f"Could not find inverse for element {i}")

            _inverse_indices[i] = j

        return _inverse_indices

    @torch.no_grad()
    def _compute_cayley_table(self) -> Tensor:
        """Computes the Cayley table (multiplication table) for the group."""
        _cayley_table = torch.zeros((self._G, self._G), dtype=torch.long)

        for i in range(self._G):
            for j in range(self._G):
                composition = self._elements[i] @ self._elements[j]

                # Find the index of the resulting element in the group
                diffs = torch.sum(
                    (self._elements - composition.unsqueeze(0)) ** 2, dim=(1, 2)
                )
                k = torch.argmin(diffs)

                if diffs[k] >= self.atol**2:
                    raise RuntimeError(
                        f"Cayley table construction failed. Product of elements {i} and {j} not found in group."
                    )

                _cayley_table[i, j] = k

        return _cayley_table

    def __repr__(self):
        return f"{self.__class__.__name__}(solid_name={self.solid_name}, G={self._G})"
