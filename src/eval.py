from datetime import datetime
import shutil
from sys import argv

from absl import app, flags
from omegaconf import OmegaConf
from ml_collections import ConfigDict

from pathlib import Path
from collections import deque

import torch
from torch import Tensor
import numpy as np
from tqdm import trange

torch.set_float32_matmul_precision("highest")

from src.utils.checkpoint import load_checkpoint
from src.utils.seed import seed_all

from src.config import load_config, load_module

FLAGS = flags.FLAGS
flags.DEFINE_string("experiment", None, "Path to experiment dir (required).")
flags.DEFINE_string("eval_config", None, "Path to rollout config YAML")
flags.mark_flag_as_required("experiment")
flags.mark_flag_as_required("eval_config")


def readout_rollout(
    rollout: list[dict[str, Tensor]], state_fields: list[str], tag: str = ""
) -> dict[list[np.ndarray]]:
    _output = {field + tag: [] for field in state_fields}

    for state in rollout:
        for field in state_fields:
            _output[field + tag].append(state[field][0].numpy(force=True))

    return _output


def compute_metrics(rollout, per_step_rollout, gt_rollout, results: dict):

    for step in range(len(rollout)):
        pred_state = rollout[step]
        per_step_pred_state = per_step_rollout[step]
        gt_state = gt_rollout[step]

        position_error = torch.mean(
            (pred_state["normal_node"] - gt_state["normal_node"]).norm(dim=-1)
        ).item()

        position_mse = torch.mean(
            (pred_state["normal_node"] - gt_state["normal_node"]) ** 2
        ).item()

        per_step_position_error = torch.mean(
            (per_step_pred_state["normal_node"] - gt_state["normal_node"]).norm(dim=-1)
        ).item()

        per_position_mse = torch.mean(
            (per_step_pred_state["normal_node"] - gt_state["normal_node"]) ** 2
        ).item()

        mean_velocity = pred_state["normal_velocity"].norm(dim=-1).mean().item()

        # compute lodging distance as the average distance between predicted and gt normal nodes at the final step

        results["position_error"].append(position_error)
        results["per_step_position_error"].append(per_step_position_error)
        results["position_mse"].append(position_mse)
        results["per_step_position_mse"].append(per_position_mse)
        results["mean_velocity"].append(mean_velocity)

    lodging_distance = torch.norm(
        pred_state["normal_node"].mean(dim=1) - gt_state["normal_node"].mean(dim=1),
    ).item()

    results["lodging_distance"] = lodging_distance
    results["mean_position_mse"] = float(np.mean(results["position_mse"]))
    results["mean_per_step_position_mse"] = float(
        np.mean(results["per_step_position_mse"])
    )

    print("mean position mse:", results["mean_position_mse"])
    print("mean per step position mse:", results["mean_per_step_position_mse"])
    print("lodging distance:", results["lodging_distance"])


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
def eval_rollout(cfg: ConfigDict, results: dict, model_path: Path, split: str):
    device = cfg.rollout.device if torch.cuda.is_available() else "cpu"

    # init datasets
    cfg.dataset.val.split = split
    dataset = load_module(cfg.dataset)(**cfg.dataset.params, **cfg.dataset.val)

    model = load_module(cfg.model)(**cfg.model.params)

    load_checkpoint(model_path / cfg.rollout.checkpoint, model)
    model = model.to(device)
    model.eval()

    for trajectory_idx in range(len(dataset.trajectories)):
        results["trajectory"][trajectory_idx] = {}
        print("--- TRAJECTORY", trajectory_idx)
        for seed in cfg.rollout.seeds:
            print("--- SEED", seed)
            seed_all(seed)

            results["trajectory"][trajectory_idx][seed] = {
                "position_error": [],
                "per_step_position_error": [],
                "position_mse": [],
                "per_step_position_mse": [],
                "mean_velocity": [],
                "lodging_distance": -1.0,
            }

            trajectory = dataset.get_trajectory(
                trajectory_idx, **cfg.rollout.trajectory
            )[cfg.rollout.start_state :]

            # setup first state
            state = {k: v.to(device) for k, v in trajectory[0].items()}
            state["normal_acceleration"] = torch.zeros(1, 1, 3)

            rollout, per_step_rollout, gt_rollout = [state], [state], []

            steps, render_gt = cfg.rollout.steps, True
            steps = min(steps, len(trajectory)) if render_gt > 0 else steps

            velocity_buffer = deque(maxlen=10)

            for step in trange(steps):  # already have first state
                if step == len(trajectory) - 1:
                    break  # no need to predict for final state (no gt next state)

                state = diffuser(state, **cfg.rollout.diffuser)

                state = model.update(state, **cfg.rollout.model_update)

                if step >= 10:
                    velocity_buffer.append(
                        state["normal_velocity"].norm(dim=-1).cpu().numpy().mean()
                    )
                    if len(velocity_buffer) == 10:
                        if np.mean(velocity_buffer) < 0.05:  # 0.4
                            print(
                                f"Early stopping at step {step} due to low velocity: {np.mean(velocity_buffer)}"
                            )
                            break

                rollout.append(state)

            for step in range(len(rollout) - 1):
                state = {k: v.to(device) for k, v in trajectory[step].items()}
                gt_rollout.append(state)

                state = model.update(state, **cfg.rollout.model_update)

                per_step_rollout.append(state)

            gt_rollout.append(
                {k: v.to(device) for k, v in trajectory[step + 1].items()}
            )

            if cfg.rollout.renderer.enable:
                renderer = load_module(cfg.rollout.renderer)
                renderer(
                    **readout_rollout(rollout, ["normal_node", "kinematic_node"]),
                    **readout_rollout(
                        gt_rollout, ["normal_node", "kinematic_node"], tag="_gt"
                    ),
                    **cfg.rollout.renderer.params,
                    save_path=None,
                )

            compute_metrics(
                rollout,
                per_step_rollout,
                gt_rollout,
                results["trajectory"][trajectory_idx][seed],
            )


def main(argv):
    cfg_eval = load_config(FLAGS.eval_config, argv)

    results = dict()

    # clean up dir if its empty except for rollout.yaml

    for model, split in zip(cfg_eval.experiment.runs, cfg_eval.experiment.splits):
        run_path = Path(cfg_eval.experiment.path) / model

        cfg = load_config(run_path / "config.yaml")
        cfg.rollout = cfg_eval

        results[model] = {
            "split": split,
            "trajectory": {},
        }

        print(f"Evaluating {model} on {split} split...")
        eval_rollout(cfg, results[model], run_path, split)

    # save results dictionary as yaml
    save_dir = Path(cfg.rollout.experiment.path) / "results"
    save_dir.mkdir(exist_ok=True, parents=True)
    OmegaConf.save(results, save_dir / f"{FLAGS.experiment}.yaml")


if __name__ == "__main__":
    app.run(main)
