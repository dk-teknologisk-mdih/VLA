#!/usr/bin/env python
"""
convert_to_rlds.py

Convert OpenTeach Franka teleop demonstrations into a native TFDS/RLDS dataset
that OpenVLA's fine-tuning pipeline can consume directly.

Input:
    The cleaned ``demo_<name>.pkl`` files produced by
    ``openteach/visualize_demo.py`` (already time-synced between the camera and
    the robot command/observation history). Each pkl contains, among others:
        - rgb_frames     (N, num_cams, H, W, 3) uint8, BGR (from cv2.VideoCapture)
        - arm_action     (N, 6) float  -> 60 Hz OSC servo command (unused here)
        - gripper_action (N,)  int in {-1, 1}   (-1 == open, +1 == close)
        - gripper_state  (N,)  float    -> physical gripper width (~0.08 == open)
        - eef_pose       (N, 4, 4) float -> absolute end-effector homogeneous pose
        - timestamp      (N,)  float

Output:
    A TFDS dataset (RLDS layout) written under ``--output-dir`` with the schema
    expected by OpenVLA:
        steps:
            observation:
                image  -> (image_size, image_size, 3) uint8 RGB, primary camera
                state  -> (8,) float32 [xyz(3), euler_rpy(3), pad(1), gripper(1)]
            action     -> (7,) float32 [dxyz(3), d_euler_rpy(3), gripper(1)]
            discount / reward / is_first / is_last / is_terminal
            language_instruction (str, provided via --instruction)
        episode_metadata:
            file_path (str)

    Each demo is first subsampled to roughly --target-hz (default 5 Hz) using
    the recorded timestamps. The action stored at each retained frame is the
    pose change the robot ACTUALLY achieved by the next retained frame
    (world-frame delta position + body-frame delta rotation), computed from
    consecutive eef_pose entries -- NOT the recorded 60 Hz servo command.
    This matches exactly how 5_control_robot_using_VLA.py integrates actions
    onto the current pose at inference time.

Example:
    python convert_to_rlds.py \
        --instruction "pick up the red block and place it in the bowl" \
        --dataset-name openteach_franka \
        --output-dir vla_data/rlds

Notes:
    - Run this with an environment that has tensorflow + tensorflow_datasets
      installed (e.g. the OpenVLA env). scipy, opencv, and numpy are also
      required.
    - This script only produces the RLDS dataset. To fine-tune with it you still
      need to register the dataset in OpenVLA's configs.py / transforms.py /
      mixtures.py (intentionally out of scope here).
"""

import argparse
import glob
import os
import pickle
import random
import shutil
import sys
import types
from pathlib import Path

import numpy as np

# Reduce TF log spam and keep the (GPU-less) build off any CUDA device.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


# Configuration populated by ``main`` and read by the TFDS builder (TFDS
# instantiates builders without arguments, so runtime options are passed here).
CONFIG = {
    "demo_paths": [],
    "instructions": [],
    "image_size": 256,
    "cam_index": 0,
    "flip_gripper": False,
    "target_hz": 5.0,
}

# Franka Panda hand max opening width (meters); used to normalize gripper state.
GRIPPER_MAX_WIDTH = 0.08
# Number of dims in the proprioceptive state and the action vector.
STATE_DIM = 8
ACTION_DIM = 7


def _ensure_easydict_importable() -> None:
    """Allow unpickling demo files that reference ``easydict.EasyDict``.

    The demo pkl embeds the deoxys ``controller_cfg`` as an ``EasyDict``. That
    package may be absent from the (TF/TFDS) environment used to build the
    dataset, which would break ``pickle.load``. We never actually use the
    config, so register a minimal, pickle-compatible stub if the real package
    is unavailable.
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


def _load_demo(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _build_steps(demo: dict) -> list:
    """Turn one loaded demo pkl into a list of RLDS step dicts.

    Frames are subsampled to ~CONFIG["target_hz"] using the recorded
    timestamps. The action at each retained frame is the pose change the robot
    ACTUALLY achieved by the next retained frame:
      - translation: world-frame delta position   (inference: target = pos + d)
      - rotation:    body-frame delta rotation    (inference: R_t = R @ dR)
    which matches exactly how 5_control_robot_using_VLA.py applies actions.
    """
    from scipy.spatial.transform import Rotation as R
    import cv2

    # Pick one instruction at random for the whole episode. This enables
    # instruction augmentation when multiple synonyms are supplied.
    instruction = random.choice(CONFIG["instructions"])
    cam_index = CONFIG["cam_index"]
    img_size = CONFIG["image_size"]
    flip_gripper = CONFIG["flip_gripper"]
    target_hz = CONFIG["target_hz"]

    num_frames = len(demo["timestamp"])
    if num_frames == 0:
        return []

    timestamps = np.asarray(demo["timestamp"], dtype=np.float64).reshape(-1)
    gripper_action = np.asarray(demo["gripper_action"]).reshape(-1)  # (N,) in {-1, 1}
    gripper_state = np.asarray(demo["gripper_state"]).reshape(-1)  # (N,)
    eef_pose = np.asarray(demo["eef_pose"], dtype=np.float64)  # (N, 4, 4)
    rgb_frames = demo["rgb_frames"]  # (N, num_cams, H, W, 3), BGR uint8

    # --- Subsample the ~60 Hz recording down to target_hz via timestamps ---
    if target_hz and target_hz > 0:
        period = 1.0 / target_hz
        keep = [0]
        for i in range(1, num_frames):
            if timestamps[i] - timestamps[keep[-1]] >= period:
                keep.append(i)
    else:
        keep = list(range(num_frames))
    keep = np.asarray(keep, dtype=np.int64)
    if keep.size < 2:
        return []

    cur, nxt = keep[:-1], keep[1:]
    num_steps = cur.size

    # --- Action: achieved delta pose between consecutive retained frames ---
    abs_pos = eef_pose[:, :3, 3]  # (N, 3)
    rot_mat = eef_pose[:, :3, :3]  # (N, 3, 3)
    # World-frame translation delta (inference: target_pos = current + dpos).
    delta_pos = abs_pos[nxt] - abs_pos[cur]
    # Body-frame rotation delta (inference: R_target = R_current @ dR).
    delta_rot = np.matmul(rot_mat[cur].transpose(0, 2, 1), rot_mat[nxt])
    delta_euler = R.from_matrix(delta_rot).as_euler("xyz")
    # Gripper: the command active at the END of the interval, i.e. the state
    # the gripper should reach by the next step. Map {-1 (open), +1 (close)}
    # -> {1.0 (open), 0.0 (close)} to match OpenVLA's convention (+1 == open).
    # Use --flip-gripper to invert.
    gripper_open = (gripper_action[nxt] < 0).astype(np.float32)
    if flip_gripper:
        gripper_open = 1.0 - gripper_open
    action = np.concatenate(
        [delta_pos, delta_euler, gripper_open[:, None]], axis=1
    ).astype(np.float32)
    assert action.shape == (num_steps, ACTION_DIM)

    # --- Proprio state: [xyz, euler (xyz), pad, gripper] (POS_EULER layout) ---
    abs_euler = R.from_matrix(rot_mat[cur]).as_euler("xyz")
    pad = np.zeros((num_steps, 1), dtype=np.float64)
    gripper_norm = np.clip(gripper_state[cur] / GRIPPER_MAX_WIDTH, 0.0, 1.0)[:, None]
    state = np.concatenate(
        [abs_pos[cur], abs_euler, pad, gripper_norm], axis=1
    ).astype(np.float32)
    assert state.shape == (num_steps, STATE_DIM)

    steps = []
    for k in range(num_steps):
        i = int(cur[k])
        # BGR -> RGB, then resize to the square resolution OpenVLA expects.
        frame = rgb_frames[i, cam_index][:, :, ::-1]
        image = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_AREA)
        steps.append(
            {
                "observation": {
                    "image": np.ascontiguousarray(image, dtype=np.uint8),
                    "state": state[k],
                },
                "action": action[k],
                "discount": np.float32(1.0),
                "reward": np.float32(1.0 if k == num_steps - 1 else 0.0),
                "is_first": bool(k == 0),
                "is_last": bool(k == num_steps - 1),
                "is_terminal": bool(k == num_steps - 1),
                "language_instruction": instruction,
            }
        )
    return steps


def _make_builder_class():
    """Create the TFDS builder class (imported lazily so --help stays fast)."""
    import tensorflow_datasets as tfds

    class OpenteachFranka(tfds.core.GeneratorBasedBuilder):
        """OpenTeach Franka teleop demonstrations in OpenVLA RLDS format."""

        VERSION = tfds.core.Version("1.0.0")
        RELEASE_NOTES = {"1.0.0": "Initial release."}

        def _info(self) -> "tfds.core.DatasetInfo":
            img = CONFIG["image_size"]
            return tfds.core.DatasetInfo(
                builder=self,
                description=(
                    "OpenTeach Franka teleoperation demonstrations converted to "
                    "the RLDS format for OpenVLA fine-tuning."
                ),
                features=tfds.features.FeaturesDict(
                    {
                        "steps": tfds.features.Dataset(
                            {
                                "observation": tfds.features.FeaturesDict(
                                    {
                                        "image": tfds.features.Image(
                                            shape=(img, img, 3),
                                            dtype=np.uint8,
                                            encoding_format="jpeg",
                                            doc="Primary camera RGB observation.",
                                        ),
                                        "state": tfds.features.Tensor(
                                            shape=(STATE_DIM,),
                                            dtype=np.float32,
                                            doc="EEF xyz (3), euler rpy (3), pad (1), gripper (1).",
                                        ),
                                    }
                                ),
                                "action": tfds.features.Tensor(
                                    shape=(ACTION_DIM,),
                                    dtype=np.float32,
                                    doc="Delta xyz (3), delta euler rpy (3), gripper (1).",
                                ),
                                "discount": tfds.features.Scalar(
                                    dtype=np.float32, doc="Discount, always 1."
                                ),
                                "reward": tfds.features.Scalar(
                                    dtype=np.float32,
                                    doc="Reward, 1 on the final step of the demo.",
                                ),
                                "is_first": tfds.features.Scalar(dtype=np.bool_),
                                "is_last": tfds.features.Scalar(dtype=np.bool_),
                                "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                                "language_instruction": tfds.features.Text(
                                    doc="Natural language task instruction."
                                ),
                            }
                        ),
                        "episode_metadata": tfds.features.FeaturesDict(
                            {
                                "file_path": tfds.features.Text(
                                    doc="Path to the source demo pkl."
                                ),
                            }
                        ),
                    }
                ),
            )

        def _split_generators(self, dl_manager):
            return {"train": self._generate_examples(CONFIG["demo_paths"])}

        def _generate_examples(self, paths):
            for path in paths:
                demo = _load_demo(path)
                steps = _build_steps(demo)
                if not steps:
                    print(f"[skip] {path} has no usable frames.")
                    continue
                # Unique, stable episode key derived from the file path.
                key = os.path.splitext(os.path.relpath(path, CONFIG["data_dir"]))[0]
                key = key.replace(os.sep, "__")
                yield key, {
                    "steps": steps,
                    "episode_metadata": {"file_path": path},
                }

    return OpenteachFranka


def _default_data_dir() -> str:
    return str(Path(__file__).resolve().parent / "vla_data" / "pickle")


def _default_output_dir() -> str:
    return str(Path(__file__).resolve().parent / "vla_data" / "rlds")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert OpenTeach Franka demos to OpenVLA RLDS (TFDS)."
    )
    parser.add_argument(
        "--data-dir",
        default=_default_data_dir(),
        help="Directory searched recursively for demo_*.pkl files "
        "(default: vla_data/pickle).",
    )
    parser.add_argument(
        "--instruction",
        help="Natural language task instruction applied to every demo. "
        "Mutually exclusive with --instruction-file.",
    )
    parser.add_argument(
        "--instruction-file",
        help="Path to a text file with one instruction (synonym) per line. "
        "A random line is chosen for each episode to augment instructions. "
        "Mutually exclusive with --instruction.",
    )
    parser.add_argument(
        "--pattern",
        default="*",
        help="Glob pattern for the demo name that follows the 'demo_' prefix "
        "and precedes the '.pkl' extension (searched recursively under "
        "--data-dir). Default '*' matches all demo_*.pkl files.",
    )
    parser.add_argument(
        "--output-dir",
        default=_default_output_dir(),
        help="TFDS data directory to write the dataset into "
        "(default: vla_data/rlds).",
    )
    parser.add_argument(
        "--dataset-name",
        default="openteach_franka",
        help="TFDS dataset name (must match [a-z0-9_]+).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Square resolution to resize camera frames to (default: 256).",
    )
    parser.add_argument(
        "--cam-index",
        type=int,
        default=0,
        help="Which camera in rgb_frames to use as the primary image (default: 0).",
    )
    parser.add_argument(
        "--target-hz",
        type=float,
        default=5.0,
        help="Subsample each demo to roughly this control frequency using the "
        "recorded timestamps before computing actions (default: 5.0). "
        "Actions become the achieved pose delta between retained frames. "
        "Set <= 0 to keep every frame.",
    )
    parser.add_argument(
        "--flip-gripper",
        action="store_true",
        help="Invert the gripper open/close mapping if your setup differs "
        "(default assumes gripper_action == -1 means open).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete any existing dataset of the same name/version first.",
    )
    args = parser.parse_args()

    if not args.dataset_name.replace("_", "").isalnum() or not args.dataset_name.islower():
        parser.error("--dataset-name must contain only lowercase letters, digits, and underscores.")

    if bool(args.instruction) == bool(args.instruction_file):
        parser.error("Provide exactly one of --instruction or --instruction-file.")

    if args.instruction_file:
        instruction_file = os.path.abspath(os.path.expanduser(args.instruction_file))
        if not os.path.isfile(instruction_file):
            parser.error(f"--instruction-file does not exist: {instruction_file}")
        with open(instruction_file, "r") as f:
            instructions = [line.strip() for line in f if line.strip()]
        if not instructions:
            parser.error(f"--instruction-file is empty: {instruction_file}")
    else:
        instructions = [args.instruction]

    _ensure_easydict_importable()

    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    if not os.path.isdir(data_dir):
        parser.error(f"--data-dir does not exist: {data_dir}")

    demo_paths = sorted(
        glob.glob(os.path.join(data_dir, "**", "demo_"+args.pattern+".pkl"), recursive=True)
    )
    if not demo_paths:
        parser.error(
            f"No demo_{args.pattern}.pkl files found under {data_dir}. "
            "Run openteach/visualize_demo.py first to generate them."
        )

    CONFIG.update(
        {
            "demo_paths": demo_paths,
            "data_dir": data_dir,
            "instructions": instructions,
            "image_size": args.image_size,
            "cam_index": args.cam_index,
            "flip_gripper": args.flip_gripper,
            "target_hz": args.target_hz,
        }
    )

    print(f"Found {len(demo_paths)} demo(s) under {data_dir}:")
    for path in demo_paths:
        print(f"  - {path}")

    builder_cls = _make_builder_class()
    # Override the TFDS-derived name with the user-provided one.
    builder_cls.name = args.dataset_name

    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    builder = builder_cls(data_dir=output_dir)

    existing = os.path.join(output_dir, args.dataset_name)
    if args.overwrite:
        if os.path.isdir(existing):
            print(f"Removing existing dataset at {existing}")
            shutil.rmtree(existing)
    elif os.path.isdir(existing):
        # TFDS's download_and_prepare() is a silent no-op when a prepared
        # dataset of the same name/version already exists. That means a stale
        # dataset from a previous run would be reused and the new demos/pattern/
        # instructions would be ignored without any warning. Fail loudly so the
        # user knows nothing was regenerated.
        parser.error(
            f"A prepared dataset already exists at {existing}.\n"
            "TFDS will NOT regenerate it, so your current --pattern / "
            "--instruction(s) would be ignored and the old data reused.\n"
            "Re-run with --overwrite to delete it and rebuild from the demos "
            "found above."
        )

    print(f"\nBuilding RLDS dataset '{args.dataset_name}' -> {output_dir} ...")
    builder.download_and_prepare()
    print("\nDone.")
    print(f"Dataset written to: {os.path.join(output_dir, args.dataset_name)}")
    print(
        "To fine-tune with OpenVLA, register this dataset in "
        "prismatic/vla/datasets/rlds/oxe/{configs,transforms,mixtures}.py and "
        f"pass --data_root_dir {output_dir} --dataset_name {args.dataset_name}."
    )


if __name__ == "__main__":
    main()
