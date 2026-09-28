from typing import Callable
import torch
from torch import nn, Tensor
from torch.nn import functional as F


from src.models.modules.batched_graph_platoformer.platociever import (
    KNNPlatonicPerciever,
)
from src.models.modules.platoformer.platociever import PlatonicPerciever
from src.models.modules.normalize import Normalizer
from src.utils.node import NodeType

from torch_cluster import fps, radius


class PlatocieverClotEntry(nn.Module):
    __constants__ = [
        "mode",
        "integration_order",
        "scale",
        "mises_scale",
        "normalize",
        "output_dim",
    ]
    __modes__: dict = {
        "signal": 1,
        "signal+velocity": 2,
        "signal+mesh": 2,
        "signal+velocity+mesh": 3,
    }

    def __init__(
        self,
        scale: float = 1.0,
        mises_scale: float = 1.0,
        output_dim: int = 0,
        mode: str = "signal",
        normalize: bool = False,
        integration_order: int = 2,
        num_knn_neighbors: int = 0,
        **kwargs,
    ):
        assert mode.lower() in list(self.__modes__)
        assert integration_order in [1, 2]

        super().__init__()

        self.scale = 1 if normalize or scale <= 0 else scale
        self.mises_scale = mises_scale
        self.normalize = normalize
        self.output_dim = output_dim

        self.mode = mode.lower()
        self.input_dim_vec = self.__modes__[self.mode]
        self.integration_order = integration_order

        self.pos_normalizer = Normalizer(3)
        self.feature_normalization = Normalizer(2)
        self.vector_normalization = Normalizer(self.input_dim_vec * 3)
        self.output_normalizer = Normalizer(3)
        self.mises_normalizer = Normalizer(1)

        self.num_knn_neighbors = num_knn_neighbors

        if num_knn_neighbors > 0:
            self.model = KNNPlatonicPerciever(
                input_dim_vec=self.input_dim_vec,
                output_dim=output_dim,
                num_neighbors=num_knn_neighbors,
                **kwargs,
            )
        else:
            self.model = PlatonicPerciever(
                input_dim_vec=self.input_dim_vec, output_dim=output_dim, **kwargs
            )

    def _unique_radius(
        self,
        x: Tensor,
        y: Tensor,
        r: float,
        batch_x: Tensor | None = None,
        batch_y: Tensor | None = None,
        max_num_neighbors: int = 1024,
    ) -> Tensor:
        """
        Find all elements in x that are within the radius of any element in y.
        """

        targets = radius(
            y,
            x,
            r=r,
            batch_x=batch_x,
            batch_y=batch_y,
            max_num_neighbors=max_num_neighbors,
        )[0]

        return torch.unique_consecutive(targets)

    def _build_vectors(
        self,
        signal: Tensor,
        velocity: Tensor | None = None,
        mesh: Tensor | None = None,
    ) -> Tensor:
        if self.mode == "signal":
            return torch.cat((signal,), dim=-1)

        if self.mode == "signal+velocity":
            return torch.cat((signal, velocity * self.scale), dim=-1)

        if self.mode == "signal+mesh":
            return torch.cat((signal, mesh * self.scale), dim=-1)

        if self.mode == "signal+velocity+mesh":
            return torch.cat((signal, velocity * self.scale, mesh * self.scale), dim=-1)

    def _forward(
        self,
        nodes: Tensor,
        node_types: Tensor,
        node_signal: Tensor,
        query_nodes: Tensor,
        query_node_types: Tensor,
        query_signal: Tensor,
        proximity_mask: Tensor,
        node_velocities: Tensor | None = None,
        query_velocities: Tensor | None = None,
        node_mesh: Tensor | None = None,
        query_mesh: Tensor | None = None,
        **_,
    ):
        pos = nodes * self.scale
        query_pos = query_nodes * self.scale

        x = F.one_hot(node_types, 2).float()
        query_x = F.one_hot(query_node_types, 2).float()

        vec = self._build_vectors(node_signal, node_velocities, node_mesh)
        query_vec = self._build_vectors(query_signal, query_velocities, query_mesh)

        if self.normalize:
            pos = self.pos_normalizer(nodes, self.training)
            query_pos = self.pos_normalizer(query_nodes, self.training)
            x: Tensor = self.feature_normalization(x, self.training)
            query_x: Tensor = self.feature_normalization(query_x, self.training)
            vec: Tensor = self.vector_normalization(vec, self.training)
            query_vec: Tensor = self.vector_normalization(query_vec, self.training)

        pred_mises, pred_gradient = self.model(
            x=x,
            pos=pos,
            query_x=query_x,
            query_pos=query_pos,
            mask=proximity_mask,
            vec=vec.unflatten(-1, (self.input_dim_vec, 3)),
            query_vec=query_vec.unflatten(-1, (self.input_dim_vec, 3)),
            # downsample_mask=node_types == NodeType.NORMAL,
        )

        return pred_mises, pred_gradient[..., 0, :]

    def forward(
        self,
        nodes: Tensor,
        node_types: Tensor,
        query_nodes: Tensor,
        node_signal: Tensor,
        query_targets: Tensor,
        query_mises_targets: Tensor,
        query_node_types: Tensor,
        query_signal: Tensor,
        proximity_mask: Tensor | None = None,
        node_velocities: Tensor | None = None,
        query_velocities: Tensor | None = None,
        node_mesh: Tensor | None = None,
        query_mesh: Tensor | None = None,
        **_,
    ) -> dict[str, Tensor]:
        pred_mises, pred_gradient = self._forward(
            nodes,
            node_types,
            node_signal,
            query_nodes,
            query_node_types,
            query_signal,
            proximity_mask=proximity_mask,
            node_velocities=node_velocities,
            query_velocities=query_velocities,
            node_mesh=node_mesh,
            query_mesh=query_mesh,
        )

        if self.integration_order == 1:
            query_targets = query_targets

        if self.integration_order == 2:  # convert target velocity to acceleration
            query_targets = query_targets - query_velocities

        if self.normalize:
            query_targets = self.output_normalizer(query_targets, self.training)
            query_mises_targets = self.mises_normalizer(
                query_mises_targets, self.training
            )
        else:
            query_targets = query_targets * self.scale
            query_mises_targets = query_mises_targets * self.mises_scale

        if self.output_dim == 1:
            return dict(
                predictions=pred_gradient,
                targets=query_targets,
                mises_predictions=pred_mises,
                mises_targets=query_mises_targets,
            )

        return dict(predictions=pred_gradient, targets=query_targets)

    @staticmethod
    @torch.no_grad
    def compute_signal_from_centerline(
        nodes: Tensor, centerline_nodes: Tensor, centerline_signal: Tensor
    ):
        batch_idx = torch.arange(nodes.shape[0], device=nodes.device)
        idx = torch.cdist(nodes, centerline_nodes).argmin(-1)

        return (
            centerline_signal[batch_idx, idx],
            centerline_nodes[batch_idx, idx],
            centerline_signal[batch_idx, idx],
        )

    @torch.no_grad
    def integrator(
        self,
        nodes: Tensor,
        node_types: Tensor,
        node_signal: Tensor,
        query_nodes: Tensor,
        query_node_types: Tensor,
        query_signal: Tensor,
        proximity_mask: Tensor,
        node_velocities: Tensor | None = None,
        query_velocities: Tensor | None = None,
        node_mesh: Tensor | None = None,
        query_mesh: Tensor | None = None,
        **_,
    ) -> Tensor:

        pred_mises, pred_gradients = self._forward(
            nodes,
            node_types,
            node_signal,
            query_nodes,
            query_node_types,
            query_signal,
            proximity_mask=proximity_mask,
            node_velocities=node_velocities,
            query_velocities=query_velocities,
            node_mesh=node_mesh,
            query_mesh=query_mesh,
        )

        if self.normalize:
            pred_gradients = self.output_normalizer.inverse(pred_gradients)
            pred_mises = self.mises_normalizer.inverse(pred_mises)
        else:
            pred_gradients = pred_gradients / self.scale
            pred_mises = pred_mises / self.mises_scale

        if self.integration_order == 1:
            return pred_gradients, pred_mises

        if self.integration_order == 2:
            return query_velocities + pred_gradients, pred_mises

    @torch.no_grad
    def compute_gradients(
        self,
        query_nodes: Tensor,
        query_velocities: Tensor,
        query_mesh: Tensor | None,
        kinematic_nodes: Tensor,
        centerline_nodes: Tensor,
        centerline_signal: Tensor,
        num_kinematic: int = 512,
        num_normal: int = 1536,
        use_fps_sampling: str = "uni",
        proximity_radius: float = 2.0,
    ) -> dict[str, Tensor]:
        device = kinematic_nodes.device

        #
        # prepare model inputs
        #

        # find all kinematic nodes that are in proximity of the updated queries
        proximity_idx = self._unique_radius(
            kinematic_nodes.view(-1, 3),
            query_nodes.view(-1, 3),
            r=proximity_radius,
        )

        kinematic_node_idx = proximity_idx[
            torch.randperm(proximity_idx.shape[0], device=device)[:num_kinematic]
        ]
        num_sampled_kinematic = kinematic_node_idx.shape[0]
        # sample new encoding nodes from the queries
        # NOTE: assumes num query > num normal encoding nodes (no reason this should not be the case)
        if use_fps_sampling:
            normal_node_idx = fps(
                query_nodes.view(-1, 3),
                ratio=1.05 * (num_normal / query_nodes.shape[1]),
            )[:num_normal]
        else:
            normal_node_idx = torch.randperm(query_nodes.shape[1], device=device)[
                :num_normal
            ]

        # new encoding nodes and velocities
        nodes = torch.zeros(
            1, num_kinematic + num_normal, 3, dtype=query_nodes.dtype, device=device
        )

        # sample encoding nodes
        nodes[:, :num_sampled_kinematic] = kinematic_nodes[:, kinematic_node_idx]
        nodes[:, num_kinematic:] = query_nodes[:, normal_node_idx]

        # construct node types
        node_types = torch.empty(
            1, num_kinematic + num_normal, dtype=torch.long, device=device
        )
        node_types[:, :num_kinematic] = NodeType.KINEMATIC
        node_types[:, num_kinematic:] = NodeType.NORMAL
        query_node_types = torch.full(
            query_nodes.shape[:-1], NodeType.NORMAL, dtype=torch.long, device=device
        )

        # construct proximity mask
        proximity_mask = torch.ones(
            1, num_kinematic + num_normal, dtype=bool, device=device
        )
        proximity_mask[:, num_sampled_kinematic:num_kinematic] = False

        # compute query and node signals
        node_signal = torch.empty_like(nodes)

        query_signal, *_ = self.compute_signal_from_centerline(
            query_nodes, centerline_nodes, centerline_signal
        )
        kinematic_node_signal, *_ = self.compute_signal_from_centerline(
            nodes[:, :num_sampled_kinematic], centerline_nodes, centerline_signal
        )

        node_signal[:, :num_sampled_kinematic] = kinematic_node_signal
        # masked nodes have no signal
        node_signal[:, num_sampled_kinematic:num_kinematic] = 0
        node_signal[:, num_kinematic:] = query_signal[:, normal_node_idx]

        if query_mesh is not None:
            node_mesh = torch.empty_like(nodes)
            node_mesh[:, :num_kinematic] = 0
            node_mesh[:, num_kinematic:] = query_mesh[:, normal_node_idx]
        else:
            node_mesh = None

        # compute node velocities
        if query_velocities is not None:
            node_velocities = torch.empty_like(nodes)
            node_velocities[:, :num_kinematic] = 0
            node_velocities[:, num_kinematic:] = query_velocities[:, normal_node_idx]
        else:
            node_velocities = None

        #
        # predict gradients
        #
        mises, gradients = self._forward(
            nodes,
            node_types,
            node_signal,
            query_nodes,
            query_node_types,
            query_signal,
            proximity_mask=proximity_mask,
            node_velocities=node_velocities,
            query_velocities=query_velocities,
            node_mesh=node_mesh,
            query_mesh=query_mesh,
        )

        # denormalize output
        if self.normalize:
            gradients = self.output_normalizer.inverse(gradients)
            mises = self.mises_normalizer.inverse(mises)
        else:
            mises /= self.mises_scale  # scale for mises during training
            gradients /= self.scale

        return gradients, mises

    @torch.no_grad
    def euler(
        self,
        query_nodes: Tensor,
        query_velocities: Tensor,
        query_mesh: Tensor | None,
        kinematic_nodes: Tensor,
        centerline_nodes: Tensor,
        centerline_signal: Tensor,
        num_kinematic: int,
        num_normal: int,
        use_fps_sampling: bool = False,
        proximity_radius: float = 2.0,
    ):
        gradients, mises = self.compute_gradients(
            query_nodes=query_nodes,
            query_velocities=query_velocities,
            query_mesh=query_mesh,
            kinematic_nodes=kinematic_nodes,
            centerline_nodes=centerline_nodes,
            centerline_signal=centerline_signal,
            num_kinematic=num_kinematic,
            num_normal=num_normal,
            use_fps_sampling=use_fps_sampling,
            proximity_radius=proximity_radius,
        )

        # in case we model simple time step function
        if self.integration_order == 1:
            return query_nodes + gradients, gradients, mises

        # in case we model ODE
        velocity = query_velocities + gradients

        return query_nodes + velocity, velocity, gradients, mises

    @torch.no_grad
    def verlet(
        self,
        query_nodes: Tensor,
        query_velocities: Tensor,
        query_mesh: Tensor | None,
        kinematic_nodes: Tensor,
        centerline_nodes: Tensor,
        centerline_signal: Tensor,
        num_kinematic: int,
        num_normal: int,
        use_fps_sampling: bool = False,
        proximity_radius: float = 2.0,
    ):
        a1, m1 = self.compute_gradients(
            query_nodes=query_nodes,
            query_velocities=query_velocities,
            query_mesh=query_mesh,
            kinematic_nodes=kinematic_nodes,
            centerline_nodes=centerline_nodes,
            centerline_signal=centerline_signal,
            num_kinematic=num_kinematic,
            num_normal=num_normal,
            use_fps_sampling=use_fps_sampling,
            proximity_radius=proximity_radius,
        )

        pos_next = query_nodes + query_velocities + 0.5 * a1

        a2, m2 = self.compute_gradients(
            query_nodes=pos_next,
            query_velocities=query_velocities + a1,
            query_mesh=query_mesh,
            kinematic_nodes=kinematic_nodes,
            centerline_nodes=centerline_nodes,
            centerline_signal=centerline_signal,
            num_kinematic=num_kinematic,
            num_normal=num_normal,
            use_fps_sampling=use_fps_sampling,
            proximity_radius=proximity_radius,
        )

        acc = 0.5 * (a1 + a2)
        vel_next = query_velocities + 0.5 * acc
        mises = 0.5 * (m1 + m2)

        return pos_next, vel_next, acc, mises

    @torch.no_grad
    def update(
        self,
        state: dict[str, Tensor],
        solver: str = "euler",
        use_fps_sampling: bool = True,
        num_kinematic: int = 512,
        num_normal: int = 1536,
        proximity_radius: float = 2.0,
    ):
        solver = solver.lower()

        integrator: Callable = None

        if solver == "euler":
            integrator = self.euler
        elif solver == "verlet":
            integrator = self.verlet
        else:
            raise ValueError(f"Unknown solver '{solver}'.")

        pos_next, vel_next, acc, mises = integrator(
            state["normal_node"],
            state["normal_velocity"],
            state["normal_mesh"],
            state["kinematic_node"],
            state["centerline_node"],
            state["centerline_signal"],
            num_kinematic,
            num_normal,
            use_fps_sampling,
            proximity_radius,
        )

        new_state = {**state}
        new_state["normal_node"] = pos_next
        new_state["normal_velocity"] = vel_next
        new_state["normal_acceleration"] = acc
        new_state["normal_mises"] = mises

        return new_state
