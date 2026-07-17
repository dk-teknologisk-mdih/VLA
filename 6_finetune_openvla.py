#!/usr/bin/env python
"""
6_finetune_openvla.py

LoRA fine-tuning of an OpenVLA checkpoint on an RLDS dataset produced by
3_convert_data_to_rlds.py (or combine_rlds_datasets.py).

What this script does automatically:
  1. Validates that the RLDS dataset exists under --data-root-dir.
  2. Registers the dataset with OpenVLA's data-loading pipeline
     (configs.py, transforms.py, mixtures.py) if not already present.
  3. Launches fine-tuning via `torchrun openvla/vla-scripts/finetune.py`.

Assumed dataset schema (produced by 3_convert_data_to_rlds.py):
  observation.image  → primary RGB camera   ("image" key, HxWx3 uint8)
  observation.state  → 8-dim POS_EULER      [xyz(3), euler_rpy(3), pad(1), gripper(1)]
  action             → 7-dim EEF_POS        [dxyz(3), d_euler_rpy(3), gripper(1)]
  language_instruction → str (per step)

Examples:
    # Basic single-GPU run:
    python 6_finetune_openvla.py \\
        --dataset-name openteach_franka \\
        --data-root-dir vla_data/rlds \\
        --run-root-dir runs/finetune

    # Multi-GPU (4 GPUs):
    python 6_finetune_openvla.py \\
        --dataset-name openteach_franka \\
        --data-root-dir vla_data/rlds \\
        --run-root-dir runs/finetune \\
        --num-gpus 4

    # Inspect what would happen without running anything:
    python 6_finetune_openvla.py \\
        --dataset-name openteach_franka \\
        --data-root-dir vla_data/rlds \\
        --dry-run

Notes:
  - Run inside the same conda/venv environment as openvla (needs peft, transformers,
    torch, etc.).  The environment at .venv/ should already have these.
  - Fine-tuned checkpoints are saved to --run-root-dir/<experiment-id>/.
  - A dataset_statistics.json is written alongside the checkpoint; keep it next to
    the weights when running inference with 5_control_robot_using_VLA.py.
"""

import argparse
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to this script's directory)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
OPENVLA_DIR = SCRIPT_DIR / "openvla"
OXE_DIR = OPENVLA_DIR / "prismatic" / "vla" / "datasets" / "rlds" / "oxe"
FINETUNE_SCRIPT = OPENVLA_DIR / "vla-scripts" / "finetune.py"

CONFIGS_FILE = OXE_DIR / "configs.py"
TRANSFORMS_FILE = OXE_DIR / "transforms.py"
MIXTURES_FILE = OXE_DIR / "mixtures.py"


# ---------------------------------------------------------------------------
# Dataset registration helpers
# ---------------------------------------------------------------------------

def _dataset_in_configs(dataset_name: str) -> bool:
    return f'"{dataset_name}"' in CONFIGS_FILE.read_text()


def _dataset_in_transforms(dataset_name: str) -> bool:
    return f'"{dataset_name}"' in TRANSFORMS_FILE.read_text()


def _dataset_in_mixtures(dataset_name: str) -> bool:
    return f'"{dataset_name}"' in MIXTURES_FILE.read_text()


def _register_in_configs(dataset_name: str, dry_run: bool) -> None:
    """Append an entry to OXE_DATASET_CONFIGS for the given dataset name."""
    entry = textwrap.dedent(f"""
    "{dataset_name}": {{
        "image_obs_keys": {{"primary": "image", "secondary": None, "wrist": None}},
        "depth_obs_keys": {{"primary": None, "secondary": None, "wrist": None}},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    }},
    """)
    # Insert before the closing brace of OXE_DATASET_CONFIGS
    src = CONFIGS_FILE.read_text()
    # Find the last closing brace of the dict
    insert_marker = "\n}\n"
    if insert_marker not in src:
        raise RuntimeError(
            f"Could not locate the closing brace of OXE_DATASET_CONFIGS in {CONFIGS_FILE}. "
            "Please add the entry manually."
        )
    new_src = src.replace(insert_marker, f"{entry}{insert_marker}", 1)
    if dry_run:
        print(f"[dry-run] Would add to {CONFIGS_FILE}:\n{entry}")
    else:
        CONFIGS_FILE.write_text(new_src)
        print(f"  [configs.py] Registered '{dataset_name}'.")


def _register_in_transforms(dataset_name: str, dry_run: bool) -> None:
    """Add a passthrough transform function and register it."""
    fn_name = f"{dataset_name}_dataset_transform"
    # Build the function source (already correct format – no remapping needed)
    fn_src = textwrap.dedent(f"""

def {fn_name}(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"Passthrough transform for '{dataset_name}'.

    The RLDS dataset produced by 3_convert_data_to_rlds.py already stores:
      - trajectory[\"action\"]               : (T, 7) float32 EEF_POS
      - trajectory[\"language_instruction\"] : (T,)  string
    No remapping is required.
    \"\"\"
    return trajectory

""")
    # Registration entry for OXE_STANDARDIZATION_TRANSFORMS
    reg_entry = f'    "{dataset_name}": {fn_name},\n'

    src = TRANSFORMS_FILE.read_text()

    # 1. Append the function before OXE_STANDARDIZATION_TRANSFORMS dict
    insert_fn_marker = "\nOXE_STANDARDIZATION_TRANSFORMS"
    if insert_fn_marker not in src:
        raise RuntimeError(
            f"Could not locate OXE_STANDARDIZATION_TRANSFORMS in {TRANSFORMS_FILE}. "
            "Please add the transform manually."
        )
    src = src.replace(insert_fn_marker, f"{fn_src}{insert_fn_marker}", 1)

    # 2. Add to the dict before its closing brace
    close_marker = "\n}\n"
    if close_marker not in src:
        raise RuntimeError(
            f"Could not locate the closing brace of OXE_STANDARDIZATION_TRANSFORMS in {TRANSFORMS_FILE}. "
            "Please add the entry manually."
        )
    src = src.replace(close_marker, f"{reg_entry}{close_marker}", 1)

    if dry_run:
        print(f"[dry-run] Would add transform '{fn_name}' to {TRANSFORMS_FILE}.")
    else:
        TRANSFORMS_FILE.write_text(src)
        print(f"  [transforms.py] Registered '{fn_name}'.")


def _register_in_mixtures(dataset_name: str, dry_run: bool) -> None:
    """Add a single-dataset mixture entry."""
    entry = textwrap.dedent(f"""
    # === {dataset_name} ===
    "{dataset_name}": [
        ("{dataset_name}", 1.0),
    ],
""")
    src = MIXTURES_FILE.read_text()
    # Insert before the closing brace of OXE_NAMED_MIXTURES
    close_marker = "\n}\n"
    if close_marker not in src:
        raise RuntimeError(
            f"Could not locate the closing brace of OXE_NAMED_MIXTURES in {MIXTURES_FILE}. "
            "Please add the entry manually."
        )
    new_src = src.replace(close_marker, f"{entry}{close_marker}", 1)
    if dry_run:
        print(f"[dry-run] Would add mixture '{dataset_name}' to {MIXTURES_FILE}.")
    else:
        MIXTURES_FILE.write_text(new_src)
        print(f"  [mixtures.py] Registered mixture '{dataset_name}'.")


def register_dataset(dataset_name: str, dry_run: bool) -> None:
    """Register the dataset in all three OpenVLA config files if needed."""
    any_added = False

    if not _dataset_in_configs(dataset_name):
        _register_in_configs(dataset_name, dry_run)
        any_added = True
    else:
        print(f"  [configs.py]   '{dataset_name}' already registered – skipping.")

    if not _dataset_in_transforms(dataset_name):
        _register_in_transforms(dataset_name, dry_run)
        any_added = True
    else:
        print(f"  [transforms.py] '{dataset_name}' already registered – skipping.")

    if not _dataset_in_mixtures(dataset_name):
        _register_in_mixtures(dataset_name, dry_run)
        any_added = True
    else:
        print(f"  [mixtures.py]  '{dataset_name}' already registered – skipping.")

    if not any_added:
        print("  Dataset already fully registered.")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Dataset
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="TFDS dataset name (directory name under --data-root-dir, e.g. 'openteach_franka').",
    )
    parser.add_argument(
        "--data-root-dir",
        default=str(SCRIPT_DIR / "vla_data" / "rlds"),
        help="Root directory containing the RLDS dataset(s). "
             "Default: vla_data/rlds (relative to this script).",
    )

    # Model
    parser.add_argument(
        "--vla-path",
        default="openvla/openvla-7b",
        help="HuggingFace Hub model ID or path to a local fine-tuned checkpoint. "
             "Default: openvla/openvla-7b",
    )
    parser.add_argument(
        "--unnorm-key",
        default=None,
        help="Dataset statistics key used for action un-normalization at inference time. "
             "Defaults to --dataset-name. Override only if the checkpoint was trained with "
             "a different key (e.g. 'bridge_orig' for the base openvla-7b).",
    )

    # Output
    parser.add_argument(
        "--run-root-dir",
        default=str(SCRIPT_DIR / "runs" / "finetune"),
        help="Directory to save fine-tuning logs and checkpoints. "
             "Default: runs/finetune (relative to this script).",
    )
    parser.add_argument(
        "--adapter-tmp-dir",
        default=str(SCRIPT_DIR / "runs" / "adapter-tmp"),
        help="Temporary directory for LoRA adapter weights before merging.",
    )

    # Training hyper-parameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-GPU batch size. Reduce if OOM (e.g. 8 for a 48 GB GPU). Default: 16.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10_000,
        help="Maximum number of gradient steps. Default: 10 000.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=2_000,
        help="Save a checkpoint every N gradient steps. Default: 2 000.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="Peak learning rate. Default: 5e-4.",
    )
    parser.add_argument(
        "--grad-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps. Default: 1.",
    )
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=10_000,
        help="RLDS shuffle buffer size. Reduce to ~10 000 if RAM is limited. Default: 100 000.",
    )
    parser.add_argument(
        "--image-aug",
        action="store_true",
        help="Enable random-crop image augmentation during training.",
    )

    # LoRA
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA rank. Default: 32.",
    )
    parser.add_argument(
        "--use-quantization",
        action="store_true",
        help="Enable 4-bit quantization (reduces VRAM but may hurt performance).",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA and perform full fine-tuning (requires significantly more VRAM).",
    )

    # Compute
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use via torchrun DDP. Default: 1.",
    )

    # W&B
    parser.add_argument(
        "--wandb-project",
        default="openvla",
        help="Weights & Biases project name. Default: openvla.",
    )
    parser.add_argument(
        "--wandb-entity",
        default="",
        help="Weights & Biases entity (team/username). Leave empty to use your default entity.",
    )

    # Misc
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print registration changes and the torchrun command without executing anything.",
    )
    parser.add_argument(
        "--skip-register",
        action="store_true",
        help="Skip automatic dataset registration (use if already registered manually).",
    )
    parser.add_argument(
        "--save-latest-only",
        action="store_true",
        default=True,
        help="Keep only the latest checkpoint (saves disk space). Default: True.",
    )
    parser.add_argument(
        "--save-all-checkpoints",
        dest="save_latest_only",
        action="store_false",
        help="Save every checkpoint instead of only the latest.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Validate dataset name
    if not re.fullmatch(r"[a-z][a-z0-9_]*", args.dataset_name):
        sys.exit(
            f"ERROR: --dataset-name '{args.dataset_name}' must be lowercase letters, "
            "digits, and underscores, and must start with a letter."
        )

    data_root_dir = Path(args.data_root_dir).expanduser().resolve()
    dataset_dir = data_root_dir / args.dataset_name

    print(f"\n=== OpenVLA Fine-Tuning: '{args.dataset_name}' ===\n")

    # ------------------------------------------------------------------
    # 1. Validate dataset exists
    # ------------------------------------------------------------------
    if not dataset_dir.is_dir():
        sys.exit(
            f"ERROR: Dataset directory not found: {dataset_dir}\n"
            f"Run 3_convert_data_to_rlds.py first to build the RLDS dataset."
        )
    # Look for a dataset_info.json to confirm it is a valid TFDS dataset
    info_files = list(dataset_dir.rglob("dataset_info.json"))
    if not info_files:
        sys.exit(
            f"ERROR: No dataset_info.json found under {dataset_dir}. "
            "The directory does not appear to contain a valid TFDS/RLDS dataset."
        )
    print(f"[1/3] Dataset found at: {dataset_dir}")

    # ------------------------------------------------------------------
    # 2. Validate openvla submodule
    # ------------------------------------------------------------------
    if not FINETUNE_SCRIPT.is_file():
        sys.exit(
            f"ERROR: Fine-tuning script not found: {FINETUNE_SCRIPT}\n"
            "Ensure the openvla/ directory contains the full repository."
        )
    for cfg_file in (CONFIGS_FILE, TRANSFORMS_FILE, MIXTURES_FILE):
        if not cfg_file.is_file():
            sys.exit(f"ERROR: Expected OpenVLA config file not found: {cfg_file}")

    # ------------------------------------------------------------------
    # 3. Register dataset in OpenVLA config files
    # ------------------------------------------------------------------
    if args.skip_register:
        print("[2/3] Skipping dataset registration (--skip-register).")
    else:
        print("[2/3] Registering dataset with OpenVLA data pipeline ...")
        register_dataset(args.dataset_name, dry_run=args.dry_run)

    # ------------------------------------------------------------------
    # 4. Build and launch the torchrun command
    # ------------------------------------------------------------------
    run_root_dir = Path(args.run_root_dir).expanduser().resolve()
    adapter_tmp_dir = Path(args.adapter_tmp_dir).expanduser().resolve()
    unnorm_key = args.unnorm_key or args.dataset_name

    torchrun_cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone",
        "--nnodes", "1",
        f"--nproc-per-node={args.num_gpus}",
        str(FINETUNE_SCRIPT),
        f"--vla_path={args.vla_path}",
        f"--data_root_dir={data_root_dir}",
        f"--dataset_name={args.dataset_name}",
        f"--run_root_dir={run_root_dir}",
        f"--adapter_tmp_dir={adapter_tmp_dir}",
        f"--batch_size={args.batch_size}",
        f"--max_steps={args.max_steps}",
        f"--save_steps={args.save_steps}",
        f"--learning_rate={args.learning_rate}",
        f"--grad_accumulation_steps={args.grad_accumulation_steps}",
        f"--shuffle_buffer_size={args.shuffle_buffer_size}",
        f"--image_aug={args.image_aug}",
        f"--lora_rank={args.lora_rank}",
        f"--use_lora={not args.no_lora}",
        f"--use_quantization={args.use_quantization}",
        f"--save_latest_checkpoint_only={args.save_latest_only}",
        f"--wandb_project={args.wandb_project}",
    ]
    if args.wandb_entity:
        torchrun_cmd.append(f"--wandb_entity={args.wandb_entity}")

    print("\n[3/3] Fine-tuning command:")
    print("  " + " \\\n    ".join(torchrun_cmd))

    if args.dry_run:
        print("\n[dry-run] Exiting without running the command.")
        return

    print(f"\nCheckpoints will be saved to: {run_root_dir}")
    print(
        "\nIMPORTANT: After training, copy dataset_statistics.json from the checkpoint\n"
        f"directory next to your model weights and pass --unnorm-key {unnorm_key}\n"
        "when running 5_control_robot_using_VLA.py.\n"
    )

    # Create output directories so they exist before torchrun starts
    run_root_dir.mkdir(parents=True, exist_ok=True)
    adapter_tmp_dir.mkdir(parents=True, exist_ok=True)

    # Launch – inherit the current environment so the active venv/conda is used
    env = os.environ.copy()
    # Ensure the openvla package is importable when running finetune.py
    python_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{OPENVLA_DIR}{os.pathsep}{python_path}" if python_path else str(OPENVLA_DIR)
    
    # --- ADD/UPDATE THESE LINES ---
    env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("WANDB_START_METHOD", "thread")
    # env.setdefault("WANDB_MODE", "disabled")  # uncomment to disable wandb
    env.setdefault("NCCL_IB_DISABLE", "1")
    # ------------------------------

    result = subprocess.run(torchrun_cmd, env=env, cwd=str(SCRIPT_DIR))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
