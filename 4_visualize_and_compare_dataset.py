#!/usr/bin/env python
"""
visualize_dataset.py

Compare a cleaned demonstration ``.pkl`` file against the RLDS/TFDS episode that
was built from it (see ``convert_to_rlds.py``). This is a debugging tool to
verify that the conversion preserved the end-effector pose and gripper signal.

It shows, in a single interactive matplotlib window:
    1. Seven line graphs overlaying the pkl-derived and RLDS values:
       end-effector position (x, y, z), rotation (roll, pitch, yaw), and gripper.
    2. An interactive slider to select the frame/timestamp.
    3. The camera image at the selected frame, for both the pkl and the RLDS
       episode, shown side by side.

Example:
    python visualize_dataset.py \
        --pkl vla_data/pickle/demo1/demo_demo1.pkl \
        --rlds-dataset openteach_franka \
        --episode-index 0
"""

import argparse
import os
import pickle
import sys
import types

import numpy as np


# Must match convert_to_rlds.py so the two sources are directly comparable.
GRIPPER_MAX_WIDTH = 0.08

# Labels for the seven proprioceptive channels we plot.
STATE_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def _ensure_easydict_importable() -> None:
    """Allow unpickling demo files that embed ``easydict.EasyDict``.

    The demo pkl stores the deoxys ``controller_cfg`` as an ``EasyDict``. That
    package may be absent from the current environment, which would break
    ``pickle.load``. We never use the config, so register a minimal stub.
    """
    try:
        import easydict  # noqa: F401

        return
    except ImportError:
        pass

    stub = types.ModuleType("easydict")

    class EasyDict(dict):
        def __setattr__(self, name, value):
            self[name] = value
            super().__setattr__(name, value)

    stub.EasyDict = EasyDict
    sys.modules["easydict"] = stub


def _load_pkl_states(pkl_path: str, cam_index: int):
    """Load the pkl and derive (state7, images).

    Returns
    -------
    state : (N, 7) float64
        [x, y, z, roll, pitch, yaw, gripper] mirroring the RLDS state layout.
    images : (N, H, W, 3) uint8, RGB
        Camera frames for ``cam_index``.
    """
    from scipy.spatial.transform import Rotation as R

    _ensure_easydict_importable()
    with open(os.path.expanduser(pkl_path), "rb") as f:
        demo = pickle.load(f)

    eef_pos = np.asarray(demo["eef_pos"], dtype=np.float64).reshape(-1, 3)
    eef_pose = np.asarray(demo["eef_pose"], dtype=np.float64)  # (N, 4, 4)
    euler = R.from_matrix(eef_pose[:, :3, :3]).as_euler("xyz")
    gripper_state = np.asarray(demo["gripper_state"], dtype=np.float64).reshape(-1)
    gripper_norm = np.clip(gripper_state / GRIPPER_MAX_WIDTH, 0.0, 1.0)

    state = np.concatenate(
        [eef_pos, euler, gripper_norm[:, None]], axis=1
    )  # (N, 7)

    # rgb_frames: (N, num_cams, H, W, 3), BGR uint8 -> RGB for the chosen cam.
    rgb_frames = np.asarray(demo["rgb_frames"])
    images = rgb_frames[:, cam_index][:, :, :, ::-1]

    return state, images


def _get_rlds_episode_lengths(dataset_name: str, builder_dir: str, data_dir: str = None):
    """Return a list with the number of steps in each RLDS episode.

    The i-th entry is the frame count of episode i.
    """
    import tensorflow_datasets as tfds

    if builder_dir:
        sys.path.append(builder_dir)

    ds = tfds.load(dataset_name, split="train", data_dir=data_dir)

    lengths = []
    for ep in ds:
        lengths.append(int(ep["steps"].cardinality().numpy()))
    return lengths


def _select_episode_by_length(lengths, target_length: int) -> int:
    """Pick an episode index whose frame count matches ``target_length``.

    - If exactly one episode matches, return it automatically.
    - If several match, print them and prompt the user to choose.
    - If none match, print an error and exit.
    """
    matches = [i for i, n in enumerate(lengths) if n == target_length]

    if not matches:
        print(
            f"[error] No RLDS episode has {target_length} frames (to match the "
            f"pkl). Available episode lengths: "
            f"{ {i: n for i, n in enumerate(lengths)} }"
        )
        sys.exit(1)

    if len(matches) == 1:
        chosen = matches[0]
        print(
            f"Auto-selected episode {chosen} ({lengths[chosen]} frames) as the "
            f"only length match."
        )
        return chosen

    print(f"Multiple RLDS episodes match the pkl length ({target_length} frames):")
    for i in matches:
        print(f"  episode {i}: {lengths[i]} frames")

    while True:
        choice = input("Enter the episode index to use: ").strip()
        try:
            idx = int(choice)
        except ValueError:
            print("Please enter a valid integer episode index.")
            continue
        if idx in matches:
            return idx
        print(f"Please pick one of the matching indices: {matches}")


def _load_rlds_states(dataset_name: str, episode_index: int, builder_dir: str, data_dir: str = None):
    """Load the requested RLDS episode and derive (state7, images).

    Returns
    -------
    state : (N, 7) float64
        [x, y, z, roll, pitch, yaw, gripper] from observation.state[:6] + [7].
    images : (N, H, W, 3) uint8, RGB
    """
    import tensorflow_datasets as tfds

    if builder_dir:
        sys.path.append(builder_dir)

    ds = tfds.load(dataset_name, split="train", data_dir=data_dir)

    episode = None
    for i, ep in enumerate(ds):
        if i == episode_index:
            episode = ep
            break
    if episode is None:
        raise IndexError(
            f"Episode index {episode_index} out of range for dataset "
            f"'{dataset_name}'."
        )

    states = []
    images = []
    for step in episode["steps"]:
        full_state = step["observation"]["state"].numpy()  # (8,)
        # [x, y, z, roll, pitch, yaw, gripper]; index 6 is pad, 7 is gripper.
        states.append(np.concatenate([full_state[:6], full_state[7:8]]))
        images.append(step["observation"]["image"].numpy())  # already RGB

    return np.asarray(states, dtype=np.float64), np.asarray(images)


def visualize(pkl_state, pkl_images, rlds_state, rlds_images):
    import matplotlib

    # The default backend may be the non-interactive "Agg" (e.g. in headless
    # or minimal environments), which cannot show a window. Try to switch to an
    # available GUI backend so the slider is interactive.
    if matplotlib.get_backend().lower() == "agg":
        for backend in ("TkAgg", "QtAgg", "Qt5Agg", "GTK3Agg", "WXAgg"):
            try:
                matplotlib.use(backend, force=True)
                break
            except Exception:
                continue
        else:
            raise RuntimeError(
                "No interactive matplotlib backend is available. Install a GUI "
                "toolkit such as PyQt5 (pip install PyQt5) or tkinter "
                "(e.g. apt-get install python3-tk) to display the window."
            )

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    n_pkl = len(pkl_state)
    n_rlds = len(rlds_state)
    n = min(n_pkl, n_rlds)
    if n_pkl != n_rlds:
        print(
            f"[warn] Frame count mismatch: pkl has {n_pkl}, rlds has {n_rlds}. "
            f"Aligning by index up to {n} frames."
        )

    frames = np.arange(max(n_pkl, n_rlds))

    fig = plt.figure(figsize=(16, 9))
    fig.subplots_adjust(bottom=0.12, hspace=0.5, wspace=0.3)
    gs = fig.add_gridspec(3, 4)

    # --- Seven line plots (positions, rotations, gripper). ---
    cursor_lines = []
    for k, label in enumerate(STATE_LABELS):
        ax = fig.add_subplot(gs[k // 4, k % 4])
        ax.plot(frames[:n_pkl], pkl_state[:, k], label="pkl", color="C0")
        ax.plot(
            frames[:n_rlds], rlds_state[:, k], label="rlds",
            color="C1", linestyle="--",
        )
        ax.set_title(label)
        ax.grid(True)
        cursor_lines.append(ax.axvline(0, color="r"))
        if k == 0:
            ax.legend(loc="upper right", fontsize="small")

    # --- Two image axes (pkl vs rlds). ---
    ax_pkl_img = fig.add_subplot(gs[1, 3])
    ax_pkl_img.set_title("pkl image")
    ax_pkl_img.axis("off")
    pkl_im = ax_pkl_img.imshow(pkl_images[0])

    ax_rlds_img = fig.add_subplot(gs[2, 3])
    ax_rlds_img.set_title("rlds image")
    ax_rlds_img.axis("off")
    rlds_im = ax_rlds_img.imshow(rlds_images[0])

    # --- Slider to scrub through frames. ---
    ax_slider = fig.add_axes((0.15, 0.03, 0.7, 0.03))
    slider = Slider(
        ax_slider, "frame", 0, n - 1, valinit=0, valstep=1,
    )

    def update(val):
        idx = int(slider.val)
        for line in cursor_lines:
            line.set_xdata([idx, idx])
        pkl_im.set_data(pkl_images[min(idx, n_pkl - 1)])
        rlds_im.set_data(rlds_images[min(idx, n_rlds - 1)])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a demo .pkl against its RLDS episode."
    )
    parser.add_argument(
        "--pkl", required=True, help="Path to the demo_<name>.pkl file."
    )
    parser.add_argument(
        "--rlds-dataset", default="openteach_franka",
        help="Name of the RLDS/TFDS dataset (default: openteach_franka).",
    )
    parser.add_argument(
        "--rlds-builder-dir", default=None,
        help="Optional path to the RLDS dataset builder to add to sys.path.",
    )
    parser.add_argument(
        "--rlds-data-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "vla_data", "rlds"),
        help="TFDS data directory to load the RLDS dataset from "
        "(default: vla_data/rlds).",
    )
    parser.add_argument(
        "--episode-index", type=int,
        help="Which RLDS episode to compare against.",
    )
    parser.add_argument(
        "--cam-index", type=int, default=0,
        help="Which camera in the pkl rgb_frames to display (default: 0).",
    )
    args = parser.parse_args()

    print(f"Loading pkl: {args.pkl}")
    pkl_state, pkl_images = _load_pkl_states(args.pkl, args.cam_index)
    print(f"  pkl frames: {len(pkl_state)}")

    episode_index = args.episode_index
    if episode_index is None:
        print("No --episode-index given; matching RLDS episodes by frame length...")
        lengths = _get_rlds_episode_lengths(
            args.rlds_dataset, args.rlds_builder_dir, args.rlds_data_dir
        )
        episode_index = _select_episode_by_length(lengths, len(pkl_state))

    print(
        f"Loading RLDS episode {episode_index} from '{args.rlds_dataset}'..."
    )
    rlds_state, rlds_images = _load_rlds_states(
        args.rlds_dataset, episode_index, args.rlds_builder_dir, args.rlds_data_dir
    )
    print(f"  rlds frames: {len(rlds_state)}")

    visualize(pkl_state, pkl_images, rlds_state, rlds_images)


if __name__ == "__main__":
    main()
