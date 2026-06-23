# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Hydra entry point for training.

Thin wrapper around ``train.py``'s config assembly. The click CLI in ``train.py`` is
the **single source of truth** for option names and defaults: this entry point reads
those defaults, overlays the resolved YAML / command-line config on top, and hands the
result to ``train.launch_from_opts``. Both entry points therefore build identical runs.

Because the defaults come from the CLI, ``configs/config.yaml`` only has to declare the
required fields (``outdir``/``cfg``/``data``/``gpus``/``batch_gpu``) — defaults are
never duplicated and so cannot drift from the CLI.

Examples
--------
    python train_hydra.py outdir=./runs cfg=stylegan3-r cond=true \
        data=./datasets/imagenet_9to4_1024x1024_16x16.zip gpus=2 batch_gpu=320
"""

import click
import hydra
from omegaconf import DictConfig, OmegaConf

import dnnlib
import train


def _cli_defaults() -> dict:
    """Default value of every ``train.py`` click option, keyed by its Python name."""
    return {p.name: p.default for p in train.main.params if isinstance(p, click.Option)}


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Start from the CLI defaults, then overlay the resolved config. resolve=True both
    # expands interpolations and enforces mandatory (???) fields like outdir/data.
    opts = dnnlib.EasyDict(_cli_defaults())
    opts.update(OmegaConf.to_container(cfg, resolve=True))
    # Apply the same normalisation the click callback does, so --metrics accepts a
    # string, a list, or "none" identically to the CLI.
    opts.metrics = train.parse_comma_separated_list(opts.metrics)
    train.launch_from_opts(opts)


if __name__ == "__main__":
    main()
