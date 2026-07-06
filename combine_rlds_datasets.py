#!/usr/bin/env python
"""
combine_rlds_datasets.py

Combine an arbitrary number of RLDS/TFDS datasets (such as those produced by
``3_convert_data_to_rlds.py``) into a single new RLDS/TFDS dataset that
OpenVLA's fine-tuning pipeline and ``4_visualize_and_compare_dataset.py`` can
consume just like any other RLDS dataset.

How it works:
    Rather than manually stitching together tfrecord shards and metadata (which
    is brittle), this regenerates a fresh dataset. Each source dataset is opened
    with TFDS, and every episode (and every step within it) is streamed into a
    new ``GeneratorBasedBuilder`` under a single ``train`` split.

    The feature schema of the combined dataset is copied verbatim from the first
    source, so image encoding, tensor shapes, and docs are preserved without
    hardcoding any dimensions. All sources must share the exact same feature
    spec; if any differ, the script errors out (it will not resize or pad).

Input:
    One or more source dataset paths. Each path may point either at:
        - the dataset directory (e.g. ``vla_data/rlds/openteach_franka``), in
          which case the single version subdirectory is auto-detected, or
        - the versioned directory directly
          (e.g. ``vla_data/rlds/openteach_franka/1.0.0``).

Output:
    A TFDS dataset (RLDS layout) written under ``--output-dir`` with
    ``--dataset-name``, combining every episode from every source.

Example:
    python combine_rlds_datasets.py \
        vla_data/rlds/openteach_franka \
        vla_data/rlds/another_dataset \
        --dataset-name combined_franka \
        --output-dir vla_data/rlds

Notes:
    - Run this with an environment that has tensorflow + tensorflow_datasets
      installed (e.g. the OpenVLA env).
    - This only produces the RLDS dataset. To fine-tune with it you still need
      to register the dataset in OpenVLA's configs.py / transforms.py /
      mixtures.py (intentionally out of scope here).
"""

import argparse
import os
import shutil
from pathlib import Path

# Reduce TF log spam and keep the (GPU-less) build off any CUDA device.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


# Configuration populated by ``main`` and read by the TFDS builder (TFDS
# instantiates builders without arguments, so runtime options are passed here).
CONFIG = {
    # List of (source_name, builder_dir) tuples for each resolved source.
    "sources": [],
}


def _resolve_builder_dir(path: str) -> str:
    """Resolve a user-supplied source path to a TFDS versioned builder dir.

    Accepts either the dataset directory (with a single version subdir) or the
    versioned directory directly. Returns the absolute versioned dir path.
    """
    abs_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(abs_path):
        raise FileNotFoundError(f"Source path does not exist or is not a directory: {abs_path}")

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


def _to_numpy(value):
    """Recursively convert a loaded TFDS episode structure to plain python.

    - Nested dicts are converted key-by-key.
    - The ``steps`` sub-``Dataset`` (a ``tf.data.Dataset``) is materialized into
      a list of step dicts.
    - ``Text`` tensors (scalar strings) are decoded to python ``str``.
    - Everything else becomes a numpy array/scalar.
    """
    import tensorflow as tf

    if isinstance(value, dict):
        return {k: _to_numpy(v) for k, v in value.items()}

    if isinstance(value, tf.data.Dataset):
        return [_to_numpy(element) for element in value]

    if isinstance(value, tf.Tensor):
        np_value = value.numpy()
        # Decode scalar byte strings (Text feature) to utf-8 python strings.
        if isinstance(np_value, bytes):
            return np_value.decode("utf-8")
        return np_value

    return value


def _make_builder_class():
    """Create the TFDS builder class (imported lazily so --help stays fast)."""
    import tensorflow_datasets as tfds

    class CombinedRlds(tfds.core.GeneratorBasedBuilder):
        """Multiple RLDS datasets merged into a single RLDS dataset."""

        VERSION = tfds.core.Version("1.0.0")
        RELEASE_NOTES = {"1.0.0": "Initial release."}

        def _info(self) -> "tfds.core.DatasetInfo":
            # Reuse the feature spec from the first source verbatim.
            first_dir = CONFIG["sources"][0][1]
            source_builder = tfds.builder_from_directory(first_dir)
            return tfds.core.DatasetInfo(
                builder=self,
                description=(
                    "Multiple RLDS datasets combined into one for OpenVLA "
                    "fine-tuning."
                ),
                features=source_builder.info.features,
            )

        def _split_generators(self, dl_manager):
            return {"train": self._generate_examples()}

        def _generate_examples(self):
            for source_idx, (source_name, builder_dir) in enumerate(CONFIG["sources"]):
                source_builder = tfds.builder_from_directory(builder_dir)
                # Combine every split from the source into our single train split.
                for split in source_builder.info.splits:
                    ds = source_builder.as_dataset(split=split)
                    for local_idx, episode in enumerate(ds):
                        example = _to_numpy(episode)
                        # Prefix with the source index (and name/split) to
                        # guarantee unique, stable keys across all combined
                        # sources, even if two sources share the same name.
                        key = f"{source_idx:03d}__{source_name}__{split}__{local_idx:06d}"
                        yield key, example

    return CombinedRlds


def _default_output_dir() -> str:
    return str(Path(__file__).resolve().parent / "vla_data" / "rlds")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine multiple RLDS datasets into a single RLDS dataset."
    )
    parser.add_argument(
        "source_paths",
        nargs="+",
        help="One or more source RLDS dataset paths. Each may be a dataset "
        "directory (version auto-detected) or a versioned directory directly.",
    )
    parser.add_argument(
        "--output-dir",
        default=_default_output_dir(),
        help="TFDS data directory to write the combined dataset into "
        "(default: vla_data/rlds).",
    )
    parser.add_argument(
        "--dataset-name",
        default="combined_rlds",
        help="TFDS dataset name for the output (must match [a-z0-9_]+).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete any existing dataset of the same name/version first.",
    )
    args = parser.parse_args()

    if not args.dataset_name.replace("_", "").isalnum() or not args.dataset_name.islower():
        parser.error(
            "--dataset-name must contain only lowercase letters, digits, and underscores."
        )

    import tensorflow_datasets as tfds

    # Resolve every source path to a versioned builder dir and validate schemas.
    sources = []
    reference_features = None
    reference_name = None
    for path in args.source_paths:
        try:
            builder_dir = _resolve_builder_dir(path)
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))

        builder = tfds.builder_from_directory(builder_dir)
        features = builder.info.features
        # Derive a filesystem-safe source name from the dataset dir name.
        source_name = os.path.basename(os.path.dirname(builder_dir)) or builder.name
        source_name = source_name.replace(os.sep, "_")

        if reference_features is None:
            reference_features = features
            reference_name = source_name
        elif str(features) != str(reference_features):
            parser.error(
                "Feature schema mismatch: source "
                f"'{source_name}' ({builder_dir}) does not match the schema of "
                f"'{reference_name}' ({sources[0][1]}). All sources must share "
                "the exact same feature spec; this script will not resize or "
                "pad.\n\n"
                f"--- {reference_name} ---\n{reference_features}\n\n"
                f"--- {source_name} ---\n{features}"
            )

        sources.append((source_name, builder_dir))

    CONFIG["sources"] = sources

    print(f"Combining {len(sources)} source dataset(s):")
    for source_name, builder_dir in sources:
        builder = tfds.builder_from_directory(builder_dir)
        total = sum(split_info.num_examples for split_info in builder.info.splits.values())
        print(f"  - {source_name}: {total} episode(s)  [{builder_dir}]")

    builder_cls = _make_builder_class()
    # Override the TFDS-derived name with the user-provided one.
    builder_cls.name = args.dataset_name

    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    out_builder = builder_cls(data_dir=output_dir)

    if args.overwrite:
        existing = os.path.join(output_dir, args.dataset_name)
        if os.path.isdir(existing):
            print(f"Removing existing dataset at {existing}")
            shutil.rmtree(existing)

    print(f"\nBuilding combined RLDS dataset '{args.dataset_name}' -> {output_dir} ...")
    out_builder.download_and_prepare()
    print("\nDone.")
    print(f"Dataset written to: {os.path.join(output_dir, args.dataset_name)}")
    print(
        "To fine-tune with OpenVLA, register this dataset in "
        "prismatic/vla/datasets/rlds/oxe/{configs,transforms,mixtures}.py and "
        f"pass --data_root_dir {output_dir} --dataset_name {args.dataset_name}."
    )


if __name__ == "__main__":
    main()
