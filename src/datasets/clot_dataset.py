import torch
import numpy as np

from torch import Tensor
from pathlib import Path
from matplotlib import pyplot as plt
import torch.utils
import torch.utils.data
import math


from src.utils.node import NodeType
from src.utils.rotate import sample_random_rotation_matrix

from torch_cluster import fps

SPLIT_DIRS = {
    "train_dev": [1, 2],
    "val_dev": [3, 4],
    "train_full": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
    ],
    "test": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    # 5-fold cross validation
    "train_1": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "val_1": [1, 2, 3, 4],
    "train_2": [1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "val_2": [5, 6, 7, 8],
    "train_3": [1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18, 19, 20],
    "val_3": [9, 10, 11, 12],
    "train_4": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18, 19, 20],
    "val_4": [13, 14, 15, 16],
    "train_5": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    "val_5": [17, 18, 19, 20],
}


def random_rotation_matrix(max_angle: float) -> torch.Tensor:
    ax, ay, az = torch.empty(3).uniform_(-max_angle, max_angle)
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)

    R = torch.tensor(
        [
            [cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz],
            [cy * sz, cx * cz + sx * sy * sz, -cz * sx + cx * sy * sz],
            [-sy, cy * sx, cx * cy],
        ],
        dtype=torch.float32,
    )

    return R


class ClotEntryDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        num_nodes: int = 2048,
        num_normal_node_queries: int = 2048,
        mode: str = "proximity",
        element_sampling_ratio: list[float] = [9.0, 1.0],
        num_nodes_per_element: list[int] = [512, 1536],
        noise_level: float = 0.0,
        gamma: float = 1.0,
        jitter: float = 0.0,
        max_rotation_angle: float = 0.0,
        split: str = "train",
        zero_mean_normal_nodes: bool = True,
        skip_first_n_states: int = 0,
        load_mises: bool = False,
        load_velocities: bool = True,
        load_mask: bool = True,
        load_signal: bool = True,
        load_mesh_state: bool = False,
        dataset_dir: Path | str = Path("dataset"),
        preload: bool = True,
        original_dataset: bool = False,
        only_load_idx: int | None = None,
        **kwargs,
    ):
        """
        Parameters:
            task (str): If `pointcloud`, returns state as pointcloud. If `mesh`, returns state as mesh (graph).
            node_distribution (str): If `default`, mesh nodes are used. If `equidistant`, nodes are resampled to be equidistantly
                                     spaced over all elements.
            num_nodes (int): Number of points sampled for `task = pointcloud`.
            element_sample_ratio (list[float] | None): Ratio of elements to sample from different object types (e.g., [vessel, clot]).
                If None, sampling is uniform. Defaults to [0.9, 0.1].
            noise_level (float): Standard deviation of Gaussian noise to add to the point cloud (clot only).
            num_neighbors (int): Number of neighbors to consider in graph-based preprocessing. Not currently implemented.
            world_edge_radius (float): Spatial radius to use for neighborhood graph construction. Not currently implemented.
            load_centerline (bool): Whether to load vessel centerline data along with point cloud.
            one_hot_labels (bool): Whether to convert node labels to one-hot encoded vectors.
            dir (Path | str): Path to the directory containing raw simulation data.
        """
        assert mode in ["default", "proximity"]
        super().__init__()

        self.dir = (
            Path(dataset_dir) if not isinstance(dataset_dir, Path) else dataset_dir
        )

        self.element_sampling_ratio = np.asarray(element_sampling_ratio)
        self.num_nodes_per_element = (
            num_nodes_per_element  # only used for proximity sampling
        )

        self.only_load_idx = only_load_idx

        self.load_mises = load_mises
        self.load_velocities = load_velocities
        self.load_mask = load_mask
        self.load_signal = load_signal
        self.load_mesh_state = load_mesh_state

        self.original_dataset = original_dataset

        self.num_normal_node_queries = num_normal_node_queries

        self.zero_mean_normal_nodes = zero_mean_normal_nodes

        self.num_nodes = num_nodes

        self.noise_level = noise_level
        self.gamma = gamma
        self.jitter = jitter

        self.skip_first_n_states = skip_first_n_states

        self.max_rotation_angle = max_rotation_angle

        self.split = split

        self.states: list[dict] = []
        self.element_sample_ps = []
        self.centerlines: list[Tensor] = []
        self.centerline_nodes: list[Tensor] = []
        self.centerline_signal: list[Tensor] = []
        self.centerline_features: list[Tensor] = []
        self.trajectories: list[range] = []
        self.zero_means: list[Tensor] = []

        self._getitem_fn = self._getitem_proximity

        self.preload = preload

        self.collate_fn = batched_trajectory_collate

        if self.preload:
            self.load_data()

    def create_states(self, trajectory, masks=None, signals=None):
        nodes = torch.from_numpy(trajectory["nodes"]).float()
        mises = torch.from_numpy(trajectory["mises"]).float()
        node_elements = torch.from_numpy(trajectory["types"]).long()
        node_types = node_elements

        p = self.element_sampling_ratio[node_types]
        p /= p.sum()

        normal_mask = node_types == NodeType.NORMAL
        nodes_mesh = nodes[0]
        nodes_mesh = nodes_mesh - nodes_mesh[normal_mask].mean(0)

        for i in range(1, nodes.shape[0] - 1):
            if i < self.skip_first_n_states:
                continue

            self.element_sample_ps.append(p)

            zeroing_mean = (
                nodes[i][normal_mask].mean(0)
                if self.zero_mean_normal_nodes
                else torch.zeros(3, dtype=torch.float32)
            )

            self.zero_means.append(zeroing_mean)

            nodes_cur = nodes[i] - zeroing_mean
            nodes_next = nodes[i + 1] - zeroing_mean
            nodes_prev = nodes[i - 1] - zeroing_mean

            state = dict(
                nodes=nodes_cur,
                nodes_next=nodes_next,
                node_elements=node_elements,  # to distinguish between different elements
                node_types=node_types,  # normal = 0 and kinematic = 1
                # node_targets=nodes_next - nodes_cur,
            )

            state["proximity_mask"] = masks[i]

            state["node_signal"] = signals[i].float()

            state["node_mises"] = mises[i]
            state["node_mises_targets"] = mises[i + 1]

            state["nodes_prev"] = nodes_prev

            state["nodes_mesh"] = nodes_mesh

            self.states.append(state)

        if self.load_mesh_state:
            # TODO: first state of the loaded trajectory without kinematic nodes
            # needs to be passed to queries and encoding nodes
            pass

        self.num_elements = torch.unique(node_elements).shape[0]

    def load_data(self, trajectory_id: int | None = None):
        trajectories = (
            [SPLIT_DIRS[self.split][trajectory_id]]
            if trajectory_id is not None
            else SPLIT_DIRS[self.split]
        )

        if self.only_load_idx is not None:
            trajectories = [trajectories[self.only_load_idx]]

        print(
            "LOADING SPLIT = ",
            self.split,
            (
                f"ONLY LOADING IDX {self.only_load_idx}"
                if self.only_load_idx is not None
                else ""
            ),
        )

        if self.original_dataset and self.split == "train":
            trajectories = trajectories[:7]

        for trajectory_id in trajectories:
            trajectory_dir = self.dir / str(trajectory_id)
            print(trajectory_dir)
            if not trajectory_dir.is_dir():
                continue

            trajectory_start = len(self.states)

            trajectory = np.load(trajectory_dir / "trajectory_data.npz")

            if self.load_mask:
                masks = torch.from_numpy(
                    np.load(trajectory_dir / "proximity_mask_r2.npy")
                ).bool()
            else:
                masks = None

            if self.load_signal:
                signals = torch.from_numpy(np.load(trajectory_dir / "signal.npy"))
                centerline = np.load(trajectory_dir / "centerline_ordered.npz")
                self.centerline_nodes.append(
                    torch.from_numpy(centerline["coordinates"]).float()
                )
                self.centerline_signal.append(
                    torch.from_numpy(centerline["directions"]).float()
                )

            else:
                signals = None

            self.create_states(trajectory, masks=masks, signals=signals)

            trajectory_end = len(self.states)

            self.trajectories.append(range(trajectory_start, trajectory_end))

    def get_initial_rollout_state(
        self, trajectory_idx: int, equidistant_sampling: bool = False
    ):
        if not self.preload:
            self.load_data(trajectory_id=trajectory_idx)
            initial_state_idx = 5
            trajectory_idx = 0
        else:
            initial_state_idx = self.trajectories[trajectory_idx][5]

        state = self.states[initial_state_idx]

        node_types = state["node_types"]
        normal_mask = node_types == NodeType.NORMAL

        nodes = state["nodes"]

        normal_nodes = nodes[normal_mask]
        normal_nodes_prev = state["nodes"][normal_mask]
        normal_node_mesh = state["nodes_mesh"][normal_mask]

        global_kinematic_nodes = nodes[~normal_mask]

        if equidistant_sampling:
            # to ensure we always have exactly the number of queries we ask for
            ratio = 1.05 * (self.num_normal_node_queries / normal_nodes.shape[0])
            indices = fps(normal_nodes, ratio=ratio)[: self.num_normal_node_queries]
            print(indices.shape)
        else:
            indices = torch.from_numpy(
                np.random.choice(
                    normal_nodes.shape[0], self.num_normal_node_queries, replace=False
                )
            )

        query_nodes = normal_nodes[indices]
        query_node_types = node_types[normal_mask][indices]
        query_velocities = query_nodes - normal_nodes_prev[indices]
        query_mesh = normal_node_mesh[indices]

        centerline_nodes = self.centerline_nodes[trajectory_idx]
        centerline_signal = self.centerline_signal[trajectory_idx]

        return dict(
            query_nodes=query_nodes,
            query_node_types=query_node_types,
            query_velocities=query_velocities,
            query_mesh=query_mesh,
            global_kinematic_nodes=global_kinematic_nodes,
            centerline_nodes=centerline_nodes,
            centerline_signal=centerline_signal,
        )

    def _getitem_proximity(self, index: int):
        state = self.states[index]

        # get all data to build input state
        sample_num_normal = self.num_nodes_per_element[NodeType.NORMAL]
        sample_num_kinematic = self.num_nodes_per_element[NodeType.KINEMATIC]

        proximity_mask = state["proximity_mask"]
        node_types = state["node_types"]
        normal_mask = node_types == NodeType.NORMAL
        kinematic_mask = ~normal_mask
        proximity_kinematic_mask = kinematic_mask & proximity_mask
        node_signal = state["node_signal"]
        nodes = state["nodes"]

        normal_nodes = nodes[normal_mask]
        normal_signal = node_signal[normal_mask]
        normal_nodes_next = state["nodes_next"][normal_mask]
        normal_nodes_prev = state["nodes_prev"][normal_mask]
        normal_nodes_mesh = state["nodes_mesh"][normal_mask]
        normal_node_mises_targets = state["node_mises_targets"][normal_mask]

        # build encoding state with kinematic nodes
        proximity_kinematic_nodes = nodes[proximity_kinematic_mask]
        proximity_kinematic_signal = node_signal[proximity_kinematic_mask]

        kinematic_proximity_mask = torch.ones(sample_num_kinematic, dtype=bool)

        kinematic_node_types = node_types[kinematic_mask][:sample_num_kinematic]
        normal_types = node_types[normal_mask]
        normal_node_types = normal_types[:sample_num_normal]

        num_kinematic_proximity = proximity_kinematic_nodes.shape[0]

        sample_nodes, sample_signal, sample_velocities, sample_nodes_mesh = torch.zeros(
            4, sample_num_kinematic + sample_num_normal, 3, dtype=torch.float32
        )
        kinematic_idx = torch.randperm(num_kinematic_proximity)[:sample_num_kinematic]

        sample_nodes[: kinematic_idx.shape[0]] = proximity_kinematic_nodes[
            kinematic_idx
        ]
        sample_signal[: kinematic_idx.shape[0]] = proximity_kinematic_signal[
            kinematic_idx
        ]

        kinematic_proximity_mask[num_kinematic_proximity:] = False

        # indices for normal nodes and queries
        indices = torch.from_numpy(
            np.random.choice(
                normal_nodes.shape[0],
                sample_num_normal + self.num_normal_node_queries,
                replace=False,
            )
        )
        indices, query_indices = (
            indices[:sample_num_normal],
            indices[sample_num_normal:],
        )

        # prepare queries
        query_nodes = normal_nodes[query_indices]
        query_nodes_next = normal_nodes_next[query_indices]
        query_nodes_prev = normal_nodes_prev[query_indices]
        query_signal = normal_signal[query_indices]
        query_nodes_mesh = normal_nodes_mesh[query_indices]
        query_mises_targets = normal_node_mises_targets[query_indices]

        if self.noise_level > 0.0:
            noise = self.noise_level * torch.randn_like(query_nodes)
            query_nodes = query_nodes + noise

            if self.gamma > 0.0:
                query_nodes_next = query_nodes_next + self.gamma * noise

        query_node_types = normal_types[query_indices]
        # compute targets after adding noise to queries
        query_node_targets = query_nodes_next - query_nodes
        query_node_velocities = query_nodes - query_nodes_prev
        # query_node_velocities[:] = query_node_velocities.mean(0)

        # prepare input state
        sample_nodes[sample_num_kinematic:] = normal_nodes[indices]
        # normal_velocities = normal_velocities[indices]
        sample_signal[sample_num_kinematic:] = normal_signal[indices]
        sample_velocities[sample_num_kinematic:] = (
            sample_nodes[sample_num_kinematic:] - normal_nodes_prev[indices]
        )
        sample_nodes_mesh[sample_num_kinematic:] = normal_nodes_mesh[indices]

        if self.noise_level > 0.0:
            sample_nodes[sample_num_kinematic:] += self.noise_level * torch.randn(
                sample_num_normal, 3, dtype=torch.float32
            )

        if self.jitter > 0.0:
            sample_nodes[:sample_num_kinematic] += self.jitter * torch.randn(
                sample_num_kinematic, 3, dtype=torch.float32
            )

        node_types = torch.cat((kinematic_node_types, normal_node_types))
        proximity_mask = torch.cat(
            (kinematic_proximity_mask, proximity_mask[normal_mask][:sample_num_normal])
        )

        if self.max_rotation_angle != 0.0:
            Rt = (
                random_rotation_matrix(self.max_rotation_angle).T
                if self.max_rotation_angle != -1.0
                else sample_random_rotation_matrix().T
            )  # sample uniform so3 matrix if rotation angle = -1
            reflection = (np.random.randint(0, 2, (3,)) * 2 - 1).astype(np.float32)
            Rt = Rt * reflection
            sample_nodes = sample_nodes @ Rt
            sample_signal = sample_signal @ Rt
            sample_velocities = sample_velocities @ Rt
            sample_nodes_mesh = sample_nodes_mesh @ Rt
            query_nodes = query_nodes @ Rt
            query_node_targets = query_node_targets @ Rt
            query_signal = query_signal @ Rt
            query_node_velocities = query_node_velocities @ Rt
            query_nodes_mesh = query_nodes_mesh @ Rt

        return dict(
            nodes=sample_nodes,
            node_types=node_types,
            node_signal=sample_signal,
            node_velocities=sample_velocities,
            node_mesh=sample_nodes_mesh,
            query_nodes=query_nodes,
            query_node_types=query_node_types,
            query_targets=query_node_targets,
            query_signal=query_signal,
            query_velocities=query_node_velocities,
            query_mesh=query_nodes_mesh,
            query_mises_targets=query_mises_targets.view(-1, 1),
            proximity_mask=proximity_mask,
        )

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return self._getitem_fn(index)

    def get_trajectory(
        self, trajectory_idx: int, num_normal: int | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        trajectory_idx = 0 if self.only_load_idx else trajectory_idx
        max_rotation_angle = self.max_rotation_angle

        self.max_rotation_angle = 0.0

        # gather trajectory data
        trajectory_raw = self.trajectories[trajectory_idx]
        state = self.states[next(iter(trajectory_raw))]

        # initialize masks
        kinematic_mask = state["node_types"] == NodeType.KINEMATIC
        normal_mask = ~kinematic_mask

        # get kinematic and mesh nodes (these dont change over time)
        kinematic_node = state["nodes"][kinematic_mask][None]

        # get centerline data (also dont change over time)
        centerline_node = self.centerline_nodes[trajectory_idx][None]
        centerline_signal = self.centerline_signal[trajectory_idx][None]

        normal_idx = (
            torch.arange(state["nodes"][normal_mask].shape[0])
            if num_normal is None
            else torch.from_numpy(
                np.random.choice(
                    state["nodes"][normal_mask].shape[0],
                    num_normal,
                    replace=False,
                )
            )
        )

        trajectory: list[dict] = []
        for state_idx in trajectory_raw:
            state = self.states[state_idx]

            normal_node = state["nodes"][normal_mask][normal_idx]
            normal_velocity = state["nodes_next"][normal_mask][normal_idx] - normal_node
            normal_mesh = state["nodes_mesh"][normal_mask][normal_idx]
            normal_mises = state["node_mises_targets"][normal_mask][normal_idx]

            trajectory.append(
                dict(
                    normal_node=normal_node[None],
                    normal_velocity=normal_velocity[None],
                    normal_mesh=normal_mesh[None],
                    normal_mises=normal_mises[None],
                    kinematic_node=kinematic_node,
                    centerline_node=centerline_node,
                    centerline_signal=centerline_signal,
                )
            )

        self.max_rotation_angle = max_rotation_angle

        return trajectory

    @staticmethod
    def rotate_state(state, rotation):
        for key, value in state.items():
            if value.shape[-1] == 3:
                state[key] = value @ rotation.T

    def __len__(self):
        return len(self.states)


def batched_trajectory_collate(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    collate = {key: [] for key in batch[0]}

    for state in batch:
        for key, data in state.items():

            # append to the batch list
            collate[key].append(data)

    for key, data in collate.items():
        collate[key] = torch.stack(data, 0)

    return collate
