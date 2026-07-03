"""Closed-loop OpenVLA control of a Franka arm via deoxys.

This script combines:
  * `openvla/vla-scripts/example_realsense.py` (RealSense capture + OpenVLA action prediction)
  * `deoxys_control/deoxys/examples/osc_control.py` (deoxys OSC_POSE Cartesian control)

into a single loop that repeatedly:
  1. takes a picture from an Intel RealSense camera,
  2. predicts a 7-DoF action with OpenVLA,
  3. moves the robot using that action (deoxys OSC_POSE controller),
  4. repeats until `--max-steps` is reached (or Ctrl-C).

OpenVLA action layout (un-normalized, single-step *relative* command):
    action[0:3] -> delta translation (dx, dy, dz) in meters
    action[3:6] -> delta rotation    (droll, dpitch, dyaw) in radians (xyz Euler)
    action[6]   -> gripper command   (~[0, 1]; for bridge_orig: 1 = open, 0 = close)

NOTE: openvla-7b + bridge_orig was trained on a WidowX/BridgeData V2 setup. Expect
to fine-tune and to verify frame/gripper conventions before trusting it on a Franka.
"""

import argparse
import time

import numpy as np
import pyrealsense2 as rs
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R
from transformers import AutoModelForVision2Seq, AutoProcessor

from deoxys import config_root
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import transform_utils
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.log_utils import get_deoxys_example_logger

logger = get_deoxys_example_logger()

# For bridge_orig, gripper value is ~[0, 1] with 1 = open, 0 = close.
GRIPPER_OPEN_THRESHOLD = 0.5
# deoxys OSC gripper command convention: -1.0 opens, 1.0 closes.
GRIPPER_OPEN_CMD = -1.0
GRIPPER_CLOSE_CMD = 1.0

# A reasonable "home" configuration to start every episode from.
RESET_JOINT_POSITIONS = [
    0.09162008114028396,
    -0.19826458111314524,
    -0.01990020486871322,
    -2.4732269941140346,
    -0.01307073642274261,
    2.30396583422025,
    0.8480939705504309,
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    # Robot / controller.
    parser.add_argument("--interface-cfg", type=str, default="charmander.yml")
    parser.add_argument("--controller-type", type=str, default="OSC_POSE")
    parser.add_argument(
        "--steps-per-action",
        type=int,
        default=20,
        help="Number of deoxys control cycles used to realize a single VLA action.",
    )
    parser.add_argument(
        "--max-pos-delta",
        type=float,
        default=0.05,
        help="Per-step translation clip (meters) applied to each VLA action.",
    )
    parser.add_argument(
        "--max-rot-delta",
        type=float,
        default=0.2,
        help="Per-step rotation clip (radians) applied to each VLA action.",
    )
    # VLA / camera.
    parser.add_argument("--model", type=str, default="openvla/openvla-7b")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--unnorm-key", type=str, default="bridge_orig")
    parser.add_argument("--instruction", type=str, default="grasp the red block")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--control-hz", type=float, default=5.0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# RealSense camera (kept open across the loop for efficiency)
# ---------------------------------------------------------------------------
class RealSenseCamera:
    """Persistent Intel RealSense color stream returning PIL images."""

    def __init__(self, width=640, height=480, fps=30, warmup_frames=30):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        self.pipeline.start(config)
        # Discard initial frames so auto-exposure / white-balance can settle.
        for _ in range(warmup_frames):
            self.pipeline.wait_for_frames()

    def get_image(self) -> Image.Image:
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("Failed to capture a color frame from the RealSense camera.")
        color_image = np.asanyarray(color_frame.get_data())
        return Image.fromarray(color_image)

    def close(self):
        self.pipeline.stop()


# ---------------------------------------------------------------------------
# OpenVLA
# ---------------------------------------------------------------------------
def load_vla(model_name: str, device: str):
    """Load the OpenVLA processor and model onto the requested device."""
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        model_name,
        # attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    return processor, vla


def build_prompt(instruction: str) -> str:
    return f"In: What action should the robot take to {instruction}?\nOut:"


def predict_action(processor, vla, image, prompt, unnorm_key, device):
    """Run the VLA and return the predicted 7-DoF action (numpy array)."""
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return np.asarray(action, dtype=np.float64)


def split_action(action):
    """Split a 7-DoF action into (delta_pos[3], delta_euler[3], gripper_scalar)."""
    action = np.asarray(action, dtype=np.float64)
    return action[0:3], action[3:6], float(action[6])


# ---------------------------------------------------------------------------
# deoxys OSC_POSE control
# ---------------------------------------------------------------------------
def osc_move(robot_interface, controller_type, controller_cfg, target_pose, gripper_cmd, num_steps):
    """Drive the end-effector toward `target_pose` using clipped OSC deltas.

    Mirrors `examples/osc_control.py::osc_move`, but the gripper command is
    parameterized instead of hard-coded.
    """
    target_pos, target_quat = target_pose
    target_pos = np.asarray(target_pos, dtype=np.float64).flatten()

    action = None
    for _ in range(num_steps):
        current_pose = robot_interface.last_eef_pose
        current_pos = current_pose[:3, 3]
        current_rot = current_pose[:3, :3]
        current_quat = transform_utils.mat2quat(current_rot)
        if np.dot(target_quat, current_quat) < 0.0:
            current_quat = -current_quat
        quat_diff = transform_utils.quat_distance(target_quat, current_quat)
        axis_angle_diff = transform_utils.quat2axisangle(quat_diff)

        action_pos = np.clip((target_pos - current_pos) * 10, -1.0, 1.0)
        action_axis_angle = np.clip(axis_angle_diff.flatten() * 1, -0.5, 0.5)

        action = action_pos.tolist() + action_axis_angle.tolist() + [gripper_cmd]
        robot_interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg,
        )
    return action


def apply_vla_action(
    robot_interface,
    controller_type,
    controller_cfg,
    action,
    num_steps,
    max_pos_delta,
    max_rot_delta,
):
    """Interpret a 7-DoF OpenVLA action and execute it on the robot.

    The translation/rotation deltas are applied in the end-effector (body) frame,
    integrated onto the current measured pose to form a Cartesian target, which is
    then reached via `osc_move`.
    """
    delta_pos, delta_euler, gripper = split_action(action)

    # Safety: clip per-step deltas against erratic predictions.
    delta_pos = np.clip(delta_pos, -max_pos_delta, max_pos_delta)
    delta_euler = np.clip(delta_euler, -max_rot_delta, max_rot_delta)

    current_pose = robot_interface.last_eef_pose
    current_pos = current_pose[:3, 3]
    current_rot = current_pose[:3, :3]

    target_pos = current_pos + delta_pos
    # Right-multiply to apply the rotation delta in the end-effector (body) frame.
    delta_rot = R.from_euler("xyz", delta_euler).as_matrix()
    target_rot = current_rot @ delta_rot
    target_quat = transform_utils.mat2quat(target_rot)

    gripper_cmd = GRIPPER_OPEN_CMD if gripper > GRIPPER_OPEN_THRESHOLD else GRIPPER_CLOSE_CMD

    osc_move(
        robot_interface,
        controller_type,
        controller_cfg,
        (target_pos, target_quat),
        gripper_cmd,
        num_steps,
    )


def wait_for_state(robot_interface):
    """Block until the robot has streamed at least one state message."""
    while robot_interface.state_buffer_size == 0:
        logger.warning("Robot state not received yet...")
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# Main closed loop
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # --- Robot ---
    robot_interface = FrankaInterface(
        config_root + f"/{args.interface_cfg}", use_visualizer=False
    )
    controller_type = args.controller_type
    controller_cfg = get_default_controller_config(controller_type)

    wait_for_state(robot_interface)
    logger.info("Resetting to home joint configuration...")
    reset_joints_to(robot_interface, RESET_JOINT_POSITIONS)

    # --- VLA + camera ---
    logger.info(f"Loading OpenVLA model '{args.model}'...")
    processor, vla = load_vla(args.model, args.device)
    prompt = build_prompt(args.instruction)
    logger.info(f"Prompt: {prompt!r}")

    camera = RealSenseCamera(
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
    )

    period = 1.0 / args.control_hz if args.control_hz > 0 else 0.0

    try:
        for step in range(args.max_steps):
            loop_start = time.time()

            # 1. Take a picture.
            image = camera.get_image()

            # 2. Generate an action with OpenVLA.
            action = predict_action(
                processor, vla, image, prompt, args.unnorm_key, args.device
            )
            logger.info(
                f"[step {step:03d}] action="
                f"{np.array2string(action, precision=3, suppress_small=True)}"
            )

            # 3. Move the robot using the generated action.
            apply_vla_action(
                robot_interface,
                controller_type,
                controller_cfg,
                action,
                num_steps=args.steps_per_action,
                max_pos_delta=args.max_pos_delta,
                max_rot_delta=args.max_rot_delta,
            )

            # Maintain a roughly fixed control rate.
            elapsed = time.time() - loop_start
            if period and elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        logger.info("Interrupted by user; stopping.")
    finally:
        camera.close()
        robot_interface.close()


if __name__ == "__main__":
    main()
