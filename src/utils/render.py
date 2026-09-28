import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from tqdm import tqdm, trange

from src.utils.node import NodeType


def animate_rollout(
    normal_node: list[np.ndarray],
    kinematic_node: np.ndarray | list[np.ndarray],
    normal_mesh: np.ndarray | list[np.ndarray] | None = None,
    normal_mises: list[np.ndarray] | None = None,
    normal_node_gt: list[np.ndarray] | None = None,
    interval: int = 100,
    save_path: str | None = None,
    **kwargs,
):
    update_kinematic_node = type(kinematic_node) is list

    assert not update_kinematic_node or (
        len(normal_node) == len(kinematic_node)
    ), "Mismatch in number of time steps."

    num_steps = len(normal_node)

    # Set up figure and 3D axis
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(projection="3d")
    # ax.view_init(elev=45, azim=-55)  # adjust as needed
    ax.grid(False)
    ax.set_axis_off()

    # Precompute bounds for consistent axis limits
    xyz_min = 0.7 * normal_node[0].min()
    xyz_max = 0.7 * normal_node[0].max()
    ax.set_xlim(xyz_min, xyz_max)
    ax.set_ylim(xyz_min, xyz_max)
    ax.set_zlim(xyz_min, xyz_max)

    # Normalize stress values for colormap
    if normal_mises is not None:
        all_stress = np.concatenate(normal_mises)
        norm = Normalize(vmin=np.min(all_stress), vmax=np.max(all_stress))
        cmap = cm.brg  # blue -> red -> green
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.05, label="Mises stress")
    elif (
        normal_mesh is not None
    ):  # we normalize mesh node coordinates to rbg color space
        normal_mesh = normal_mesh[0] if isinstance(normal_mesh, list) else normal_mesh
        normal_mesh = normal_mesh - normal_mesh.min()
        normal_mesh /= normal_mesh.max()
        normal_mesh = np.pad(
            normal_mesh, ((0, 0), (0, 1)), "constant", constant_values=0
        )

    # Plot nodes
    normal_scatter = ax.scatter([], [], [], s=10)
    kinematic_scatter = ax.scatter(
        *(kinematic_node[0] if isinstance(kinematic_node, list) else kinematic_node).T,
        c="tab:orange",
        s=5,
        alpha=1,
        label="Kinematic",
    )
    gt_scatter = (
        ax.scatter([], [], [], c="tab:gray", s=10, label="GT", alpha=0.5)
        if normal_node_gt is not None
        else None
    )
    ax.legend()
    ax.set_aspect("equal")

    def init():
        normal_scatter._offsets3d = ([], [], [])
        if update_kinematic_node:
            kinematic_scatter._offsets3d = ([], [], [])
        if gt_scatter is not None:
            gt_scatter._offsets3d = ([], [], [])

        return normal_scatter, kinematic_scatter, gt_scatter

    pbar = iter(trange(num_steps, desc="Rendering animation"))

    def update(frame):
        if pbar is not None:
            next(pbar)
        if normal_mises is not None:
            mises = normal_mises[frame]
            normal_scatter.set_color(cmap(norm(mises)))
        elif normal_mesh is not None:
            normal_scatter.set_color(normal_mesh)

        s_normal_node = normal_node[frame]

        normal_scatter._offsets3d = (
            s_normal_node[:, 0],
            s_normal_node[:, 1],
            s_normal_node[:, 2],
        )

        if gt_scatter is not None:
            gt = normal_node_gt[frame]
            gt_scatter._offsets3d = (gt[:, 0], gt[:, 1], gt[:, 2])

        if update_kinematic_node:
            s_kinematic_node = kinematic_node[frame]
            kinematic_scatter._offsets3d = (
                s_kinematic_node[:, 0],
                s_kinematic_node[:, 1],
                s_kinematic_node[:, 2],
            )

        fig.suptitle(f"Time step {frame + 1}/{num_steps}", y=0.99)

        return normal_scatter, kinematic_scatter

    anim = FuncAnimation(
        fig, update, init_func=init, frames=num_steps, interval=interval, blit=False
    )

    if save_path:
        anim.save(save_path, writer="ffmpeg", dpi=100)
        print(f"Saved animation to {save_path}")
    else:
        pbar = None
        try:
            plt.show()
        except:
            print(f"Animation display failed!")


def animate_clot(
    normal_node: list[np.ndarray],
    kinematic_node: np.ndarray | list[np.ndarray],
    normal_mesh: list[np.ndarray] | None = None,
    normal_mises: list[np.ndarray] | None = None,
    normal_node_gt: list[np.ndarray] | None = None,
    interval: int = 100,
    save_path: str | None = None,
    **kwargs,
):
    update_kinematic_node = type(kinematic_node) is list

    assert not update_kinematic_node or (
        len(normal_node) == len(kinematic_node)
    ), "Mismatch in number of time steps."

    num_steps = len(normal_node)

    # Set up figure and 3D axis
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(projection="3d")
    # ax.view_init(elev=45, azim=-55)  # adjust as needed
    ax.grid(False)
    ax.set_axis_off()

    # Precompute bounds for consistent axis limits
    mins = np.concatenate([normal_node[0], kinematic_node[0]], axis=0).min(axis=0)
    maxs = np.concatenate([normal_node[0], kinematic_node[0]], axis=0).max(axis=0)

    center = (mins + maxs) / 2
    extents = maxs - mins

    side = extents.max()

    half = side / 2
    x_min, y_min, z_min = center - half
    x_max, y_max, z_max = center + half
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    # Normalize stress values for colormap
    if normal_mises is not None:
        all_stress = np.concatenate(normal_mises)
        norm = Normalize(vmin=np.min(all_stress), vmax=np.max(all_stress))
        cmap = cm.brg  # blue -> red -> green
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.05, label="Mises stress")
    elif (
        normal_mesh is not None
    ):  # we normalize mesh node coordinates to rbg color space
        normal_mesh = normal_mesh[0] if isinstance(normal_mesh, list) else normal_mesh
        normal_mesh = normal_mesh - normal_mesh.min()
        normal_mesh /= normal_mesh.max()
        normal_mesh = np.pad(
            normal_mesh, ((0, 0), (0, 1)), "constant", constant_values=0
        )

    # Plot nodes
    normal_scatter = ax.scatter([], [], [], s=10, c="tab:red")
    kinematic_scatter = ax.scatter(
        *(kinematic_node[0] if isinstance(kinematic_node, list) else kinematic_node).T,
        c="tab:gray",
        s=5,
        alpha=0.2,
        label="Kinematic",
    )
    gt_scatter = (
        ax.scatter([], [], [], c="tab:green", s=10, label="GT", alpha=0.5)
        if normal_node_gt is not None
        else None
    )
    ax.legend()
    ax.set_aspect("equal")

    def init():
        normal_scatter._offsets3d = ([], [], [])
        if update_kinematic_node:
            kinematic_scatter._offsets3d = ([], [], [])
        if gt_scatter is not None:
            gt_scatter._offsets3d = ([], [], [])

        return normal_scatter, kinematic_scatter, gt_scatter

    pbar = iter(trange(num_steps, desc="Rendering animation"))

    def update(frame):
        if pbar is not None:
            next(pbar)
        if normal_mises is not None:
            mises = normal_mises[frame]
            normal_scatter.set_color(cmap(norm(mises)))
        elif normal_mesh is not None:
            normal_scatter.set_color(normal_mesh)

        s_normal_node = normal_node[frame]

        normal_scatter._offsets3d = (
            s_normal_node[:, 0],
            s_normal_node[:, 1],
            s_normal_node[:, 2],
        )

        if gt_scatter is not None:
            gt = normal_node_gt[frame]
            gt_scatter._offsets3d = (gt[:, 0], gt[:, 1], gt[:, 2])

        if update_kinematic_node:
            s_kinematic_node = kinematic_node[frame]
            kinematic_scatter._offsets3d = (
                s_kinematic_node[:, 0],
                s_kinematic_node[:, 1],
                s_kinematic_node[:, 2],
            )

        fig.suptitle(f"Time step {frame + 1}/{num_steps}", y=0.99)

        return normal_scatter, kinematic_scatter

    anim = FuncAnimation(
        fig, update, init_func=init, frames=num_steps, interval=interval, blit=False
    )

    if save_path:
        anim.save(save_path, writer="ffmpeg", dpi=100)
        print(f"Saved animation to {save_path}")
    else:
        pbar = None
        try:
            plt.show()
        except:
            print(f"Animation display failed!")


def animate_average_vel_accel(
    velocity,
    acceleration,
    velocity_gt=None,
    acceleration_gt=None,
    interval=100,  # ms between frames
    save_path=None,
):
    num_steps = len(velocity)
    t = np.arange(num_steps)

    avg_velocity = [np.mean(np.linalg.norm(v, axis=1)) for v in velocity]
    avg_acceleration = [np.mean(np.linalg.norm(a, axis=1)) for a in acceleration]

    avg_velocity_gt = None
    avg_acceleration_gt = None

    if velocity_gt is not None:
        avg_velocity_gt = [np.mean(np.linalg.norm(v, axis=1)) for v in velocity_gt]
    if acceleration_gt is not None:
        avg_acceleration_gt = [
            np.mean(np.linalg.norm(a, axis=1)) for a in acceleration_gt
        ]

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # --- velocity plot ---
    (line_v,) = ax[0].plot([], [], label="Predicted", color="tab:red")
    line_v_gt = None
    if velocity_gt is not None:
        (line_v_gt,) = ax[0].plot([], [], label="Ground truth", color="tab:green")

    ax[0].set_xlim(0, num_steps - 1)
    ax[0].set_ylim(
        0,
        max(
            max(avg_velocity),
            max(avg_velocity_gt) if avg_velocity_gt else 0,
        )
        * 1.1,
    )
    ax[0].set_title("Mean velocity over time")
    ax[0].set_xlabel(r"Time in $\Delta t$")
    ax[0].set_ylabel(r"Mean velocity in mm/$\Delta t$")
    legend = ax[0].legend(loc="upper right")
    legend.set_zorder(10)  # put legend on top of time bar

    ax[0].grid()

    # --- acceleration plot ---
    (line_a,) = ax[1].plot([], [], label="Predicted", color="tab:red")
    line_a_gt = None
    if acceleration_gt is not None:
        (line_a_gt,) = ax[1].plot([], [], label="Ground truth", color="tab:green")

    ax[1].set_xlim(0, num_steps - 1)
    ax[1].set_ylim(
        0,
        max(
            max(avg_acceleration),
            max(avg_acceleration_gt) if avg_acceleration_gt else 0,
        )
        * 1.1,
    )
    ax[1].set_title("Mean acceleration over time")
    ax[1].set_xlabel(r"Time in $\Delta t$")
    ax[1].set_ylabel(r"Mean acceleration in mm/$\Delta t^2$")
    ax[1].legend()
    legend = ax[1].legend(loc="upper right")
    legend.set_zorder(10)  # put legend on top of time bar
    ax[1].grid()

    time_bar_v = ax[0].axvline(0, color="black", alpha=0.7)
    time_bar_v.set_clip_on(False)
    time_bar_a = ax[1].axvline(0, color="black", alpha=0.7)
    time_bar_a.set_clip_on(False)

    time_text_v = ax[0].text(
        0.02, 0.95, "", transform=ax[0].transAxes, ha="left", va="top"
    )
    time_text_a = ax[1].text(
        0.02, 0.95, "", transform=ax[1].transAxes, ha="left", va="top"
    )

    plt.tight_layout()

    def init():
        line_v.set_data([], [])
        line_a.set_data([], [])

        time_bar_v.set_xdata([0, 0])
        time_bar_a.set_xdata([0, 0])

        time_text_v.set_text("")
        time_text_a.set_text("")

        artists = [line_v, line_a, time_bar_v, time_bar_a, time_text_v, time_text_a]

        if velocity_gt is not None:
            line_v_gt.set_data([], [])
            artists.append(line_v_gt)

        if acceleration_gt is not None:
            line_a_gt.set_data([], [])
            artists.append(line_a_gt)

        return artists

    pbar = iter(trange(num_steps, desc="Rendering animation"))

    # --- animation update ---
    def update(frame):
        if pbar is not None:
            next(pbar)

        line_v.set_data(t[:frame], avg_velocity[:frame])
        line_a.set_data(t[:frame], avg_acceleration[:frame])

        time_bar_v.set_xdata([frame, frame])
        time_bar_a.set_xdata([frame, frame])

        time_text_v.set_text(rf"$t = {frame}\,\Delta t$")
        time_text_a.set_text(rf"$t = {frame}\,\Delta t$")

        artists = [line_v, line_a, time_bar_v, time_bar_a, time_text_v, time_text_a]

        if velocity_gt is not None:
            line_v_gt.set_data(t[:frame], avg_velocity_gt[:frame])
            artists.append(line_v_gt)

        if acceleration_gt is not None:
            line_a_gt.set_data(t[:frame], avg_acceleration_gt[:frame])
            artists.append(line_a_gt)

        return artists

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=num_steps,
        interval=interval,
        blit=True,
    )

    if save_path:
        anim.save(save_path)
        print(f"Saved animation to {save_path}")
    else:
        plt.show()


def plot_average_vel_accel(
    velocity, acceleration, velocity_gt=None, acceleration_gt=None, save_path=None
):
    num_steps = len(velocity)

    avg_velocity = [np.mean(np.linalg.norm(v, axis=1)) for v in velocity]
    avg_acceleration = [np.mean(np.linalg.norm(a, axis=1)) for a in acceleration]

    avg_velocity_gt = None
    avg_acceleration_gt = None

    if velocity_gt is not None:
        avg_velocity_gt = [np.mean(np.linalg.norm(v, axis=1)) for v in velocity_gt]
    if acceleration_gt is not None:
        avg_acceleration_gt = [
            np.mean(np.linalg.norm(a, axis=1)) for a in acceleration_gt
        ]

    _, ax = plt.subplots(2, 1, figsize=(10, 8))

    ax[0].plot(range(num_steps), avg_velocity, label="Predicted", color="tab:red")
    if velocity_gt is not None:
        ax[0].plot(
            range(num_steps), avg_velocity_gt, label="Ground truth", color="tab:green"
        )
    ax[0].set_xlim(0, num_steps - 1)
    ax[0].set_ylim(
        0,
        max(
            max(avg_velocity),
            max(avg_velocity_gt) if avg_velocity_gt else 0,
        )
        * 1.1,
    )
    ax[0].set_title("Mean velocity over Time")
    ax[0].set_xlabel(r"Time in $\Delta t$")
    ax[0].set_ylabel(r"Mean velocity in mm/$\Delta t$")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(range(num_steps), avg_acceleration, label="Predicted", color="tab:red")
    if acceleration_gt is not None:
        ax[1].plot(
            range(num_steps),
            avg_acceleration_gt,
            label="Ground truth",
            color="tab:green",
        )
    ax[1].set_xlim(0, num_steps - 1)
    ax[1].set_ylim(
        0,
        max(
            max(avg_acceleration),
            max(avg_acceleration_gt) if avg_acceleration_gt else 0,
        )
        * 1.1,
    )
    ax[1].set_title("Mean acceleration over time")
    ax[1].set_xlabel(r"Time in $\Delta t$")
    ax[1].set_ylabel(r"Mean acceleration in mm/$\Delta t^2$")
    ax[1].legend()
    ax[1].grid()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved mean velocity and acceleration plot to {save_path}")
    else:
        plt.show()


def clot_renderer(
    normal_node: list[np.ndarray],
    normal_velocity: list[np.ndarray],
    normal_acceleration: list[np.ndarray],
    kinematic_node: np.ndarray | list[np.ndarray],
    normal_mesh: list[np.ndarray] | None = None,
    normal_mises: list[np.ndarray] | None = None,
    normal_node_gt: list[np.ndarray] | None = None,
    normal_velocity_gt: list[np.ndarray] | None = None,
    normal_acceleration_gt: list[np.ndarray] | None = None,
    interval: int = 100,
    save_path: str | None = None,
    **kwargs,
):
    """
    renders clot animation and average velocity and acceleration over time with or without ground truth.
    """
    print("Rendering clot animation...")
    animate_clot(
        normal_node,
        kinematic_node,
        normal_mesh,
        normal_mises,
        normal_node_gt,
        interval,
        save_path / "anim_clot.gif" if save_path else None,
    )

    print("Rendering average velocity and acceleration animation and plot...")
    animate_average_vel_accel(
        normal_velocity,
        normal_acceleration,
        normal_velocity_gt,
        normal_acceleration_gt,
        interval,
        save_path / "anim_vel_accel.gif" if save_path else None,
    )
    plot_average_vel_accel(
        normal_velocity,
        normal_acceleration,
        normal_velocity_gt,
        normal_acceleration_gt,
        save_path / "plot_vel_accel.png" if save_path else None,
    )
