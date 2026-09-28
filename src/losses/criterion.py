import torch
from torch import nn, Tensor


class MSELoss(nn.Module):
    def __init__(self, boundary_weight: float = 1.0, radius: float = 0.1, **kwargs):
        super().__init__()

        self.radius = radius
        self.boundary_weight = boundary_weight

    def near_boundary_loss(
        self, predictions, targets, kinematic_pos: Tensor, query_pos: Tensor
    ) -> Tensor:
        with torch.no_grad():
            # compute distance from each query point to the nearest kinematic point
            distances = torch.cdist(query_pos, kinematic_pos)

            # nodes that are within radius of kinematic nodes
            mask = (distances <= self.radius).sum(-1).bool()

        return self.mse_loss(predictions[mask], targets[mask])

    def mse_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        loss = ((predictions - targets) ** 2).sum(-1)
        loss = loss.mean()

        return loss

    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        kinematic_pos: Tensor,
        query_pos: Tensor,
    ):
        node_loss = self.mse_loss(predictions, targets)
        boundary_loss = self.boundary_weight * self.near_boundary_loss(
            predictions, targets, kinematic_pos, query_pos
        )

        loss = node_loss + boundary_loss

        return loss, dict(
            node_loss=node_loss.item(), boundary_loss=boundary_loss.item()
        )


class BoundaryWeightedMSELoss(nn.Module):
    __constants__ = ["radius", "max_weight"]

    def __init__(self, radius, max_weight):
        super().__init__()

        self.radius = radius
        self.max_weight = max_weight

    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        queries: Tensor,
        boundary_nodes: Tensor,
    ):
        with torch.no_grad():
            weights = (
                self.radius - (torch.cdist(queries, boundary_nodes)).min(dim=-1)[0]
            ) * (self.max_weight - 1) + 1

        loss = weights * ((predictions - targets) ** 2).sum(-1)
        loss = loss.mean()

        return loss


class MSEClotLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        mises_predictions: Tensor | None = None,
        mises_targets: Tensor | None = None,
    ):
        loss = ((predictions - targets) ** 2).sum(-1)
        loss = loss.mean()

        if mises_predictions is not None:
            mises_loss = ((mises_predictions - mises_targets) ** 2).mean()
            loss = loss + mises_loss

        return loss
