from collections import deque
from datetime import datetime
import shutil

from absl import app, flags
from omegaconf import OmegaConf
from ml_collections import ConfigDict

from pathlib import Path

import torch
from torch import Tensor
import numpy as np
from tqdm import trange

torch.set_float32_matmul_precision("medium")

from src.utils.checkpoint import load_checkpoint
from src.utils.seed import seed_all

from src.config import load_config, load_module

FLAGS = flags.FLAGS
flags.DEFINE_string("experiment", None, "Path to experiment dir (required).")
flags.DEFINE_string(
    "rollout_config", "configs/rollout.yaml", "Path to rollout config YAML"
)
flags.mark_flag_as_required("experiment")
flags.DEFINE_bool("show_only", False, "Only show existing results without rendering.")
flags.DEFINE_bool(
    "save_rollout", False, "Whether to save rollout data as .npz in output dir."
)
flags.DEFINE_bool("per_step", False, "Whether to compute per-step rollout")


def readout_rollout(
    rollout: list[dict[str, Tensor]], state_fields: list[str]
) -> dict[list[np.ndarray]]:
    _output = {field: [] for field in state_fields}

    for state in rollout:
        for field in state_fields:
            _output[field].append(state[field][0].numpy(force=True))

    return _output


def readout_trajectory(trajectory: list[dict[str, Tensor]]) -> dict[list[np.ndarray]]:
    _output = {
        "normal_node_gt": [],
        "normal_velocity_gt": [],
        "normal_acceleration_gt": [],
        "normal_mises_gt": [],
    }

    for i in range(len(trajectory)):
        s_cur = trajectory[i]
        _output["normal_velocity_gt"].append(
            s_cur["normal_velocity"][0].numpy(force=True)
        )
        _output["normal_node_gt"].append(s_cur["normal_node"][0].numpy(force=True))

        # note that state contains TARGET mises, so the targets from the previous
        # state are the ground truth for the current state - same for acceleration
        if i != 0:
            s_prev = trajectory[i - 1]
            _output["normal_acceleration_gt"].append(
                (s_cur["normal_velocity"] - s_prev["normal_velocity"])[0].numpy(
                    force=True
                )
            )

            if "normal_mises" in s_prev:
                _output["normal_mises_gt"].append(
                    s_prev["normal_mises"][0].numpy(force=True)
                )

    # insert (zero) acceleration mises forces for very first state
    _output["normal_acceleration_gt"].insert(
        0, np.zeros_like(_output["normal_acceleration_gt"][0])
    )
    if "normal_mises" in s_prev:
        _output["normal_mises_gt"].insert(
            0, np.zeros_like(_output["normal_mises_gt"][0])
        )

    return _output


def setup(argv) -> tuple[ConfigDict, Path]:
    experiment = Path(FLAGS.experiment)
    experiment_cfg_path = experiment / "config.yaml"

    if not experiment_cfg_path.exists():
        raise FileNotFoundError(
            f"Experiment config at '{experiment_cfg_path}' does not exist."
        )

    cfg = load_config(experiment_cfg_path)
    cfg.rollout = load_config(FLAGS.rollout_config, argv)

    # overwrite dataset settings
    if "dataset" in cfg.rollout:
        cfg.dataset = cfg.rollout.dataset

    if "tag" in cfg.rollout and cfg.rollout.tag:
        tag = cfg.rollout.tag
    else:
        tag = None

    output_dir = (
        experiment
        / "rollout"
        / f"{cfg.rollout.split}_t{cfg.rollout.trajectory.trajectory_idx}{f'_{tag}' if tag is not None else ''}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg.rollout.to_dict(), output_dir / "rollout.yaml")

    return cfg, output_dir


def diffuser(
    state: dict[str, Tensor],
    noise_level: float = 0.0,
    gamma: float = 0.0,
    jitter: float = 0.0,
    __cache: list[Tensor] = [],
):
    if noise_level > 0.0:
        noise = torch.randn_like(state["normal_node"]) * noise_level
        state["normal_node"] = state["normal_node"] + noise
        state["normal_velocity"] = state["normal_velocity"] + gamma * noise

    if jitter > 0.0:
        if len(__cache) == 0:
            __cache.append(state["kinematic_node"].clone())

        jitter = torch.randn_like(__cache[-1]) * jitter
        state["kinematic_node"] = __cache[-1] + jitter

    return state


@torch.no_grad
def run_rollout(cfg: ConfigDict, output_dir: Path):
    print(cfg)

    device = cfg.rollout.device if torch.cuda.is_available() else "cpu"

    model = load_module(cfg.model)(**cfg.model.params)

    cfg.dataset.val.split = cfg.rollout.split
    dataset = load_module(cfg.dataset)(
        **cfg.dataset.params,
        **cfg.dataset.val,
        only_load_idx=cfg.rollout.trajectory.trajectory_idx,
    )

    epoch = load_checkpoint(Path(FLAGS.experiment) / cfg.rollout.checkpoint, model)
    model = model.to(device)
    model.eval()

    print(f"Initalized model checkpoint - {epoch = }")
    print(f"SETTING SEED", cfg.rollout.seed)
    seed_all(cfg.rollout.seed)

    trajectory = dataset.get_trajectory(**cfg.rollout.trajectory)[
        cfg.rollout.start_state :
    ]
    trajectory_gt = {}

    # setup first state
    state = {k: v.to(device) for k, v in trajectory[0].items()}
    state["normal_acceleration"] = torch.zeros(1, 1, 3)

    rollout = [state]
    velocity_buffer = deque(maxlen=10)

    steps, render_gt = cfg.rollout.steps, cfg.rollout.renderer.render_gt
    steps = min(steps, len(trajectory)) if render_gt > 0 else steps
    for step in trange(steps - 1):  # already have first stat
        state = diffuser(state, **cfg.rollout.diffuser)

        state = model.update(state, **cfg.rollout.model_update)
        mean_velocity = state["normal_velocity"].norm(dim=-1).mean()

        if (
            cfg.rollout.max_velocity is not None
            and mean_velocity > cfg.rollout.max_velocity
        ):
            state["normal_velocity"] /= mean_velocity
            state["normal_velocity"] *= cfg.rollout.max_velocity

        if cfg.rollout.min_velocity is not None and step >= 10:
            velocity_buffer.append(mean_velocity.cpu().numpy())
            if len(velocity_buffer) == 10:
                if np.mean(velocity_buffer) < 0.05:
                    print(
                        f"Early stopping at step {step} due to low velocity: {np.mean(velocity_buffer)}"
                    )
                    break

        rollout.append(state)

        if FLAGS.per_step:
            state = {k: v.to(device) for k, v in trajectory[step + 1].items()}

    rollout = readout_rollout(rollout, cfg.rollout.state_fields)

    if render_gt:
        trajectory_gt = readout_trajectory(trajectory)

    if FLAGS.save_rollout:
        if "normal_node_gt" in trajectory_gt:
            rollout["normal_node_gt"] = trajectory_gt["normal_node_gt"]
            rollout["normal_mises_gt"] = trajectory_gt["normal_mises_gt"]
        else:
            print("NO GT TRAJECTORY AVAILABLE")

        np.savez(
            output_dir / ("per_step_rollout.npz" if FLAGS.per_step else "rollout.npz"),
            **rollout,
        )
        exit()

    renderer = load_module(cfg.rollout.renderer)
    renderer(
        **rollout,
        **trajectory_gt,
        **cfg.rollout.renderer.params,
        save_path=output_dir if not FLAGS.show_only else None,
    )


def main(argv):
    cfg, output_dir = setup(argv)

    # clean up dir if its empty except for rollout.yaml
    try:
        run_rollout(cfg, output_dir)
    finally:
        if all(f.name == "rollout.yaml" for f in output_dir.iterdir()):
            print(f"Cleaning up empty rollout directory: {output_dir}")
            shutil.rmtree(output_dir)


if __name__ == "__main__":
    app.run(main)
