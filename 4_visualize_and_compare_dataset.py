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

# Labels for the seven action dimensions (delta EEF command).
ACTION_LABELS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


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


def _load_all_rlds_actions(dataset_name: str, builder_dir: str, data_dir: str = None):
    """Load actions from every step across all RLDS episodes.

    Returns
    -------
    all_actions : (N_total, 7) float64
        Every action vector in the dataset, concatenated across all episodes.
    sample_images : list of (H, W, 3) uint8
        One image per episode (the first frame), for a visual sanity check.
    n_episodes : int
    episode_lengths : list[int]
    """
    import tensorflow_datasets as tfds

    if builder_dir:
        sys.path.append(builder_dir)

    ds = tfds.load(dataset_name, split="train", data_dir=data_dir)

    all_actions = []
    sample_images = []
    episode_lengths = []

    for ep in ds:
        ep_actions = []
        first_img = None
        for step in ep["steps"]:
            action = step["action"].numpy()  # (7,)
            ep_actions.append(action)
            if first_img is None:
                first_img = step["observation"]["image"].numpy()
        all_actions.extend(ep_actions)
        episode_lengths.append(len(ep_actions))
        if first_img is not None:
            sample_images.append(first_img)

    return (
        np.asarray(all_actions, dtype=np.float64),
        sample_images,
        len(episode_lengths),
        episode_lengths,
    )


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


def visualize_rlds_actions(
    dataset_name: str,
    all_actions,
    sample_images,
    n_episodes: int,
    episode_lengths,
    stats_json: str = None,
):
    """Show action distribution histograms and sample images for an RLDS dataset.

    Optionally overlays q01/q99 boundaries from a ``dataset_statistics.json``
    file (e.g. the one saved alongside a fine-tuned OpenVLA checkpoint) so you
    can see whether the training data fills the normalization range.
    """
    import json
    import matplotlib

    if matplotlib.get_backend().lower() == "agg":
        for backend in ("TkAgg", "QtAgg", "Qt5Agg", "GTK3Agg", "WXAgg"):
            try:
                matplotlib.use(backend, force=True)
                break
            except Exception:
                continue

    import matplotlib.pyplot as plt

    # Load optional norm stats to overlay q01/q99 boundaries.
    norm_q01 = norm_q99 = None
    if stats_json and os.path.isfile(stats_json):
        with open(stats_json) as f:
            stats = json.load(f)
        # dataset_statistics.json has one top-level key = dataset name.
        key = list(stats.keys())[0]
        action_stats = stats[key]["action"]
        norm_q01 = np.array(action_stats["q01"])
        norm_q99 = np.array(action_stats["q99"])
        print(f"Loaded norm stats from '{stats_json}' (key='{key}')")

    n_total = len(all_actions)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(
        f"Action distributions — {dataset_name}\n"
        f"{n_episodes} episodes · {n_total} steps  "
        f"(lengths: min={min(episode_lengths)}, "
        f"max={max(episode_lengths)}, "
        f"mean={np.mean(episode_lengths):.1f})",
        fontsize=11,
    )

    for i, (ax, label) in enumerate(zip(axes.flat[:7], ACTION_LABELS)):
        col = all_actions[:, i]
        ax.hist(col, bins=60, color="steelblue", edgecolor="none", alpha=0.85)
        ax.axvline(col.mean(), color="orange", linewidth=1.5, label=f"mean={col.mean():.4f}")
        ax.axvline(np.percentile(col, 1), color="red", linewidth=1, linestyle="--",
                   label=f"q01={np.percentile(col, 1):.4f}")
        ax.axvline(np.percentile(col, 99), color="green", linewidth=1, linestyle="--",
                   label=f"q99={np.percentile(col, 99):.4f}")
        if norm_q01 is not None:
            ax.axvline(norm_q01[i], color="red", linewidth=2, linestyle=":",
                       label=f"stats q01={norm_q01[i]:.4f}")
            ax.axvline(norm_q99[i], color="green", linewidth=2, linestyle=":",
                       label=f"stats q99={norm_q99[i]:.4f}")
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("action value")
        ax.set_ylabel("steps")
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.3)

    # Last cell: grid of sample images (one per episode, up to 8).
    ax_img = axes.flat[7]
    ax_img.axis("off")
    n_show = min(len(sample_images), 8)
    if n_show > 0:
        cols = 4
        rows = (n_show + cols - 1) // cols
        inner = ax_img.inset_axes([0, 0, 1, 1])
        inner.axis("off")
        for j, img in enumerate(sample_images[:n_show]):
            r, c = divmod(j, cols)
            sub = inner.inset_axes(
                [c / cols, 1 - (r + 1) / rows, 1 / cols, 1 / rows]
            )
            sub.imshow(img)
            sub.set_title(f"ep {j}", fontsize=6, pad=1)
            sub.axis("off")
        ax_img.set_title("First frame / episode", fontsize=8)

    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize an RLDS dataset's action distributions, or compare a "
            "demo .pkl against its RLDS episode.\n\n"
            "RLDS-only mode (no --pkl):\n"
            "  python 4_visualize_and_compare_dataset.py --rlds-dataset red_block_in_cardboard\n\n"
            "Compare mode:\n"
            "  python 4_visualize_and_compare_dataset.py "
            "--pkl vla_data/pickle/demo1/demo.pkl --rlds-dataset red_block_in_cardboard"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pkl", default=None,
        help="Path to the demo_<name>.pkl file. "
             "Omit to visualize the RLDS dataset action distributions only.",
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
        "--stats-json", default=None,
        help="Path to a dataset_statistics.json file (e.g. from a fine-tuned "
             "OpenVLA checkpoint) to overlay q01/q99 normalization boundaries "
             "on the action histograms.",
    )
    parser.add_argument(
        "--episode-index", type=int,
        help="Which RLDS episode to compare against (compare mode only).",
    )
    parser.add_argument(
        "--cam-index", type=int, default=0,
        help="Which camera in the pkl rgb_frames to display (default: 0).",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # RLDS-only mode: show action distribution histograms across all episodes.
    # -----------------------------------------------------------------------
    if args.pkl is None:
        print(f"No --pkl provided. Loading all actions from '{args.rlds_dataset}'...")
        all_actions, sample_images, n_ep, ep_lengths = _load_all_rlds_actions(
            args.rlds_dataset, args.rlds_builder_dir, args.rlds_data_dir
        )
        print(
            f"  Loaded {len(all_actions)} steps across {n_ep} episodes "
            f"(lengths: {ep_lengths})"
        )
        # Print per-dimension statistics to stdout for quick inspection.
        print(f"\n{'Dim':<8} {'mean':>10} {'std':>10} {'q01':>10} {'q99':>10} {'min':>10} {'max':>10}")
        for i, label in enumerate(ACTION_LABELS):
            col = all_actions[:, i]
            print(
                f"{label:<8} {col.mean():>10.5f} {col.std():>10.5f} "
                f"{np.percentile(col, 1):>10.5f} {np.percentile(col, 99):>10.5f} "
                f"{col.min():>10.5f} {col.max():>10.5f}"
            )
        visualize_rlds_actions(
            args.rlds_dataset,
            all_actions,
            sample_images,
            n_ep,
            ep_lengths,
            stats_json=args.stats_json,
        )
        return

    # -----------------------------------------------------------------------
    # Compare mode: pkl vs RLDS side-by-side.
    # -----------------------------------------------------------------------
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
