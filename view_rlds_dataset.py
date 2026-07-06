#!/usr/bin/env python
"""
view_rlds_dataset.py

Inspect and verify an RLDS/TFDS dataset such as those produced by
``3_convert_data_to_rlds.py`` or merged by ``combine_rlds_datasets.py``.

This is a read-only viewer. It never modifies the dataset. Its purpose is to let
you confirm, at a glance, that everything ended up the way you expect:
    - which source files were merged into the dataset (episode_metadata.file_path),
    - what language instruction each episode carries,
    - how many steps each episode has,
    - dataset-wide statistics and the full feature schema, and
    - a set of basic sanity checks that flag likely problems.

It prints:
    1. A dataset overview (name, version, episode/step counts, feature schema,
       and the auto-detected image feature keys).
    2. A per-episode table: index | steps | unique instruction(s) | file_path.
    3. A verification section listing any warnings (empty episodes, NaN/Inf in
       state or action, out-of-range gripper values, misplaced is_first/is_last/
       is_terminal flags, missing instructions, and duplicate file paths).

Then, unless ``--no-gui`` is given, it opens an interactive matplotlib window to
scrub through frames of a chosen episode, showing the camera image alongside
line plots of the proprioceptive state channels, with an episode selector so you
can browse the whole dataset without restarting.

Input:
    A single dataset path, which may point either at:
        - the dataset directory (e.g. ``vla_data/rlds/openteach_franka``), in
          which case the single version subdirectory is auto-detected, or
        - the versioned directory directly
          (e.g. ``vla_data/rlds/openteach_franka/1.0.0``).

Example:
    python view_rlds_dataset.py vla_data/rlds/openteach_franka
    python view_rlds_dataset.py vla_data/rlds/combined_franka --no-gui
    python view_rlds_dataset.py vla_data/rlds/openteach_franka --episode-index 2

Notes:
    - Run this with an environment that has tensorflow + tensorflow_datasets
      installed (e.g. the OpenVLA env). numpy is also required; matplotlib is
      only needed for the interactive window (skip it with --no-gui).
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Reduce TF log spam and keep the (GPU-less) load off any CUDA device. These
# must be set before tensorflow is imported. Level "3" hides INFO/WARNING/ERROR
# C++ logs (including the harmless "no CUDA-capable device" and dataset-cache
# messages); disabling oneDNN removes the oneDNN startup notice.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def _silence_absl_logging() -> None:
    """Quiet absl's python logging, which TF uses for its INFO/WARNING lines."""
    try:
        from absl import logging as absl_logging

        absl_logging.set_verbosity(absl_logging.ERROR)
    except Exception:
        pass


# Must match 3_convert_data_to_rlds.py so gripper values are interpreted the
# same way when we range-check them.
GRIPPER_MAX_WIDTH = 0.08

# Labels for the proprioceptive channels we plot: state is
# [x, y, z, roll, pitch, yaw, pad, gripper]; index 6 (pad) is dropped for plots.
STATE_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def _resolve_builder_dir(path: str) -> str:
    """Resolve a user-supplied path to a TFDS versioned builder dir.

    Accepts either the dataset directory (with a single version subdir) or the
    versioned directory directly. Returns the absolute versioned dir path.

    Mirrors the resolution logic used by combine_rlds_datasets.py so the two
    tools accept exactly the same kinds of paths.
    """
    abs_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(abs_path):
        raise FileNotFoundError(
            f"Dataset path does not exist or is not a directory: {abs_path}"
        )

    # A versioned builder dir contains the TFDS metadata files directly.
    if os.path.isfile(os.path.join(abs_path, "dataset_info.json")):
        return abs_path

    # Otherwise treat it as a dataset dir and look for version subdirectories
    # (directories that themselves contain dataset_info.json).
    version_dirs = sorted(
        entry.path
        for entry in os.scandir(abs_path)
        if entry.is_dir()
        and os.path.isfile(os.path.join(entry.path, "dataset_info.json"))
    )
    if not version_dirs:
        raise FileNotFoundError(
            f"No TFDS dataset found at {abs_path}. Expected either a "
            "dataset_info.json here or a version subdirectory containing one."
        )
    if len(version_dirs) > 1:
        raise ValueError(
            f"Multiple version subdirectories found under {abs_path}: "
            f"{version_dirs}. Point directly at the versioned directory you want."
        )
    return version_dirs[0]


def _detect_image_keys(features) -> list:
    """Return the names of every Image feature under steps -> observation.

    Introspects the TFDS feature spec (no shapes are hardcoded) so the viewer
    works with single- or multi-camera datasets alike.
    """
    import tensorflow_datasets as tfds

    try:
        observation = features["steps"]["observation"]
    except (KeyError, TypeError):
        return []

    image_keys = []
    for key in observation.keys():
        if isinstance(observation[key], tfds.features.Image):
            image_keys.append(key)
    return sorted(image_keys)


def _decode_text(value) -> str:
    """Decode a possibly-bytes scalar string tensor value to a python str."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _state_to_plot_channels(state: np.ndarray) -> np.ndarray:
    """Map a raw state vector to the 7 plotted channels.

    The RLDS state layout is [x, y, z, roll, pitch, yaw, pad, gripper]. We drop
    the pad (index 6) so the columns line up with STATE_LABELS. Shorter states
    are returned unchanged (best-effort for non-standard schemas).
    """
    if state.shape[-1] >= 8:
        return np.concatenate([state[..., :6], state[..., 7:8]], axis=-1)
    return state


def _scan_dataset(builder, image_key: Optional[str]):
    """Iterate every episode once, collecting summary info and check data.

    Returns
    -------
    episodes : list of dict
        One entry per episode with keys: index, num_steps, instructions (sorted
        list of unique strings), file_path (str or None), and warnings (list of
        str) produced by the per-episode sanity checks.
    """
    import tensorflow_datasets as tfds

    # Disable TFDS auto-caching so early-exit iteration (in the GUI) doesn't emit
    # "calling iterator did not fully read the dataset being cached" warnings.
    read_config = tfds.ReadConfig(try_autocache=False)

    episodes = []
    global_index = 0
    for split in builder.info.splits:
        ds = builder.as_dataset(split=split, read_config=read_config)
        for episode in ds:
            info = _scan_episode(episode, global_index, split, image_key)
            episodes.append(info)
            global_index += 1
    return episodes


def _scan_episode(episode, index: int, split: str, image_key: Optional[str]) -> dict:
    """Collect summary + sanity-check data for a single episode."""
    warnings = []

    # --- episode_metadata.file_path (provenance). ---
    file_path = None
    metadata = episode.get("episode_metadata", {})
    if "file_path" in metadata:
        file_path = _decode_text(metadata["file_path"].numpy())

    # --- Walk the steps once, gathering everything we need. ---
    instructions = set()
    states = []
    actions = []
    first_flags = []
    last_flags = []
    terminal_flags = []

    for step in episode["steps"]:
        if "language_instruction" in step:
            instructions.add(_decode_text(step["language_instruction"].numpy()))

        observation = step.get("observation", {})
        if "state" in observation:
            states.append(observation["state"].numpy())
        if "action" in step:
            actions.append(step["action"].numpy())

        if "is_first" in step:
            first_flags.append(bool(step["is_first"].numpy()))
        if "is_last" in step:
            last_flags.append(bool(step["is_last"].numpy()))
        if "is_terminal" in step:
            terminal_flags.append(bool(step["is_terminal"].numpy()))

    num_steps = len(states) if states else len(actions)
    if not states and not actions:
        # Fall back to counting steps directly if this dataset has neither.
        num_steps = int(episode["steps"].cardinality().numpy())

    states = np.asarray(states) if states else np.empty((0, 0))
    actions = np.asarray(actions) if actions else np.empty((0, 0))

    # --- Sanity checks. ---
    if num_steps == 0:
        warnings.append("episode has 0 steps")

    if states.size and not np.all(np.isfinite(states)):
        warnings.append("state contains NaN/Inf")
    if actions.size and not np.all(np.isfinite(actions)):
        warnings.append("action contains NaN/Inf")

    # Gripper is the last channel of both state and action in this schema and
    # is expected to be normalized to [0, 1].
    if states.size and states.shape[-1] >= 8:
        gripper = states[:, -1]
        if gripper.size and (gripper.min() < -1e-6 or gripper.max() > 1 + 1e-6):
            warnings.append(
                f"state gripper out of [0,1] "
                f"(min={gripper.min():.3f}, max={gripper.max():.3f})"
            )
    if actions.size and actions.shape[-1] >= 7:
        gripper_action = actions[:, -1]
        if gripper_action.size and (
            gripper_action.min() < -1e-6 or gripper_action.max() > 1 + 1e-6
        ):
            warnings.append(
                f"action gripper out of [0,1] "
                f"(min={gripper_action.min():.3f}, max={gripper_action.max():.3f})"
            )

    # Boundary flags: is_first only on step 0; is_last/is_terminal only on last.
    if first_flags:
        expected_first = [i == 0 for i in range(len(first_flags))]
        if first_flags != expected_first:
            warnings.append("is_first not set exclusively on the first step")
    if last_flags:
        expected_last = [i == len(last_flags) - 1 for i in range(len(last_flags))]
        if last_flags != expected_last:
            warnings.append("is_last not set exclusively on the last step")
    if terminal_flags:
        expected_terminal = [
            i == len(terminal_flags) - 1 for i in range(len(terminal_flags))
        ]
        if terminal_flags != expected_terminal:
            warnings.append("is_terminal not set exclusively on the last step")

    # Instruction presence.
    non_empty = {ins for ins in instructions if ins.strip()}
    if not non_empty:
        warnings.append("missing/empty language instruction")

    return {
        "index": index,
        "split": split,
        "num_steps": num_steps,
        "instructions": sorted(instructions),
        "file_path": file_path,
        "warnings": warnings,
    }


def _print_overview(builder, image_keys, image_key):
    info = builder.info
    total_episodes = sum(s.num_examples for s in info.splits.values())

    print("=" * 78)
    print("RLDS DATASET OVERVIEW")
    print("=" * 78)
    print(f"  name         : {info.name}")
    print(f"  version      : {info.version}")
    print(f"  splits       : {', '.join(info.splits.keys()) or '(none)'}")
    print(f"  episodes     : {total_episodes}")
    for split_name, split_info in info.splits.items():
        print(f"      - {split_name}: {split_info.num_examples} episode(s)")
    if info.description:
        first_line = info.description.strip().splitlines()[0]
        print(f"  description  : {first_line}")
    if image_keys:
        print(f"  image keys   : {', '.join(image_keys)}  (showing '{image_key}')")
    else:
        print("  image keys   : (none detected)")
    print()
    print("  feature schema:")
    for line in str(info.features).splitlines():
        print(f"    {line}")
    print()


def _print_episode_table(episodes):
    print("=" * 78)
    print("EPISODES")
    print("=" * 78)

    # Render each instruction fully (no truncation) and size the column to the
    # widest one so the file_path column still lines up.
    def _instruction_text(ep):
        return " | ".join(ep["instructions"]) if ep["instructions"] else "(none)"

    instr_header = "instruction(s)"
    instr_width = max(
        [len(instr_header)] + [len(_instruction_text(ep)) for ep in episodes]
    )

    header = f"{'idx':>4}  {'steps':>6}  {instr_header:<{instr_width}}  file_path"
    print(header)
    print("-" * len(header))
    for ep in episodes:
        instruction = _instruction_text(ep)
        file_path = ep["file_path"] if ep["file_path"] is not None else "(no metadata)"
        print(
            f"{ep['index']:>4}  {ep['num_steps']:>6}  "
            f"{instruction:<{instr_width}}  {file_path}"
        )
    print()


def _print_verification(episodes):
    print("=" * 78)
    print("VERIFICATION")
    print("=" * 78)

    total_steps = sum(ep["num_steps"] for ep in episodes)
    all_instructions = sorted(
        {ins for ep in episodes for ins in ep["instructions"] if ins.strip()}
    )

    print(f"  total episodes : {len(episodes)}")
    print(f"  total steps    : {total_steps}")
    print(f"  unique instructions ({len(all_instructions)}):")
    for ins in all_instructions:
        count = sum(1 for ep in episodes if ins in ep["instructions"])
        print(f"      - [{count:>3} ep] {ins}")

    # Duplicate file_path detection (provenance collisions).
    seen = {}
    duplicates = {}
    for ep in episodes:
        fp = ep["file_path"]
        if fp is None:
            continue
        if fp in seen:
            duplicates.setdefault(fp, [seen[fp]]).append(ep["index"])
        else:
            seen[fp] = ep["index"]

    print()
    per_episode_warnings = [(ep["index"], w) for ep in episodes for w in ep["warnings"]]
    if not per_episode_warnings and not duplicates:
        print("  No problems detected. \u2713")
        print()
        return

    print("  WARNINGS:")
    for index, warning in per_episode_warnings:
        print(f"      - episode {index}: {warning}")
    for fp, indices in duplicates.items():
        print(f"      - duplicate file_path across episodes {indices}: {fp}")
    print()


def _load_episode_frames(builder, episode_index: int, image_key: Optional[str]):
    """Load one episode's (states7, images, instruction) for the GUI."""
    import tensorflow_datasets as tfds

    # Disable auto-caching so breaking out early doesn't warn about a partially
    # read cached dataset.
    read_config = tfds.ReadConfig(try_autocache=False)

    global_index = 0
    for split in builder.info.splits:
        ds = builder.as_dataset(split=split, read_config=read_config)
        for episode in ds:
            if global_index == episode_index:
                return _extract_episode_frames(episode, image_key)
            global_index += 1
    raise IndexError(f"Episode index {episode_index} out of range.")


def _extract_episode_frames(episode, image_key: Optional[str]):
    states = []
    images = []
    instructions = set()
    for step in episode["steps"]:
        observation = step.get("observation", {})
        if "state" in observation:
            states.append(_state_to_plot_channels(observation["state"].numpy()))
        if image_key and image_key in observation:
            images.append(observation[image_key].numpy())
        if "language_instruction" in step:
            instructions.add(_decode_text(step["language_instruction"].numpy()))

    states = np.asarray(states, dtype=np.float64) if states else np.empty((0, 7))
    images = np.asarray(images) if images else None
    instruction = " | ".join(sorted(instructions)) if instructions else "(none)"
    return states, images, instruction


def _select_interactive_backend() -> bool:
    """Try to switch matplotlib to an interactive backend.

    Returns True if an interactive backend is active, False if only the
    non-interactive Agg backend is available (headless environment).
    """
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        return True
    for backend in ("TkAgg", "QtAgg", "Qt5Agg", "GTK3Agg", "WXAgg"):
        try:
            matplotlib.use(backend, force=True)
            return True
        except Exception:
            continue
    return False


def visualize(builder, episodes, image_key: Optional[str], start_index: int):
    """Interactive browser: scrub frames and switch episodes."""
    if not _select_interactive_backend():
        print(
            "[warn] No interactive matplotlib backend is available; skipping the "
            "GUI. Install PyQt5 (pip install PyQt5) or tkinter "
            "(apt-get install python3-tk), or re-run with --no-gui to silence "
            "this."
        )
        return

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    num_episodes = len(episodes)

    fig = plt.figure(figsize=(16, 9))
    fig.subplots_adjust(bottom=0.16, top=0.92, hspace=0.5, wspace=0.35)
    gs = fig.add_gridspec(3, 4)

    # Line plots for the seven state channels.
    axes = []
    lines = []
    cursor_lines = []
    for k, label in enumerate(STATE_LABELS):
        ax = fig.add_subplot(gs[k // 4, k % 4])
        (line,) = ax.plot([], [], color="C0")
        ax.set_title(label)
        ax.grid(True)
        axes.append(ax)
        lines.append(line)
        cursor_lines.append(ax.axvline(0, color="r"))

    # Image axis.
    ax_img = fig.add_subplot(gs[1:, 3])
    ax_img.axis("off")
    image_artist = None

    # Frame slider.
    ax_frame = fig.add_axes((0.15, 0.06, 0.6, 0.03))
    frame_slider = Slider(ax_frame, "frame", 0, 1, valinit=0, valstep=1)

    # Episode selector (only meaningful if there is more than one episode).
    ax_episode = fig.add_axes((0.15, 0.02, 0.6, 0.03))
    episode_slider = Slider(
        ax_episode,
        "episode",
        0,
        max(num_episodes - 1, 1),
        valinit=start_index,
        valstep=1,
    )

    state = {"states": None, "images": None, "instruction": "", "num_frames": 0}

    def load_episode(episode_index: int):
        states, images, instruction = _load_episode_frames(
            builder, episode_index, image_key
        )
        state["states"] = states
        state["images"] = images
        state["instruction"] = instruction
        state["num_frames"] = max(len(states), 0 if images is None else len(images))

        frames = np.arange(len(states))
        for k in range(len(STATE_LABELS)):
            if states.size and states.shape[1] > k:
                lines[k].set_data(frames, states[:, k])
                axes[k].relim()
                axes[k].autoscale_view()

        nonlocal image_artist
        if images is not None and len(images):
            if image_artist is None:
                image_artist = ax_img.imshow(images[0])
            else:
                image_artist.set_data(images[0])
            ax_img.set_title(f"{image_key}")
        else:
            ax_img.set_title(f"{image_key} (no image)")

        n = max(state["num_frames"], 1)
        frame_slider.valmax = n - 1
        frame_slider.ax.set_xlim(0, n - 1)
        if frame_slider.val > n - 1:
            frame_slider.set_val(0)

        meta = episodes[episode_index]
        file_path = meta["file_path"] or "(no metadata)"
        fig.suptitle(
            f"episode {episode_index}/{num_episodes - 1}  |  "
            f"{state['num_frames']} frames  |  {state['instruction']}\n{file_path}",
            fontsize=10,
        )
        update_frame(0)

    def update_frame(val):
        if state["states"] is None:
            return
        idx = int(frame_slider.val)
        for line in cursor_lines:
            line.set_xdata([idx, idx])
        if state["images"] is not None and len(state["images"]):
            clamped = min(idx, len(state["images"]) - 1)
            if image_artist is not None:
                image_artist.set_data(state["images"][clamped])
        fig.canvas.draw_idle()

    def on_episode_change(val):
        load_episode(int(episode_slider.val))
        fig.canvas.draw_idle()

    frame_slider.on_changed(update_frame)
    episode_slider.on_changed(on_episode_change)

    load_episode(start_index)
    plt.show()


def _default_dataset_path() -> str:
    return str(Path(__file__).resolve().parent / "vla_data" / "rlds" / "openteach_franka")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and verify an RLDS/TFDS dataset."
    )
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default=_default_dataset_path(),
        help="Path to the RLDS dataset. May be a dataset directory (version "
        "auto-detected) or a versioned directory directly "
        "(default: vla_data/rlds/openteach_franka).",
    )
    parser.add_argument(
        "--image-key",
        default=None,
        help="Which observation image feature to display in the GUI "
        "(default: the first auto-detected Image feature).",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Episode to open first in the GUI (default: 0).",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Print the summary and verification only; do not open the window.",
    )
    args = parser.parse_args()

    import tensorflow_datasets as tfds

    _silence_absl_logging()

    try:
        builder_dir = _resolve_builder_dir(args.dataset_path)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    builder = tfds.builder_from_directory(builder_dir)
    print(f"Loaded dataset from: {builder_dir}\n")

    image_keys = _detect_image_keys(builder.info.features)
    if args.image_key is not None:
        if args.image_key not in image_keys:
            parser.error(
                f"--image-key '{args.image_key}' is not an Image feature. "
                f"Available image keys: {image_keys or '(none)'}"
            )
        image_key = args.image_key
    else:
        image_key = image_keys[0] if image_keys else None

    _print_overview(builder, image_keys, image_key)

    print("Scanning episodes ...\n")
    episodes = _scan_dataset(builder, image_key)

    _print_episode_table(episodes)
    _print_verification(episodes)

    if args.no_gui:
        return

    if not episodes:
        print("[warn] No episodes to display; skipping the GUI.")
        return

    start_index = args.episode_index
    if start_index < 0 or start_index >= len(episodes):
        print(
            f"[warn] --episode-index {start_index} out of range "
            f"(0..{len(episodes) - 1}); starting at 0."
        )
        start_index = 0

    visualize(builder, episodes, image_key, start_index)


if __name__ == "__main__":
    main()
