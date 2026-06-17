# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Hydra entry point for training.

This is a thin wrapper around the existing config-assembly logic in ``train.py``:
it loads a YAML config (configs/config.yaml + optional overrides), converts it to the
same ``opts`` mapping the click CLI produces, and hands it to ``train.build_config`` /
``train.launch_from_opts``. Both entry points therefore produce identical runs.

Examples
--------
    python train_hydra.py outdir=./runs data=./data/ffhq16.zip gpus=8 batch_gpu=8
    python train_hydra.py +experiment=ffhq16_stem outdir=./runs data=./data/ffhq16.zip
"""

import hydra
from omegaconf import DictConfig, OmegaConf

import dnnlib
import train


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Convert the (possibly nested) Hydra config into the flat EasyDict of CLI-style
    # options that train.build_config expects. resolve=True expands any interpolations.
    opts = dnnlib.EasyDict(OmegaConf.to_container(cfg, resolve=True))
    train.launch_from_opts(opts)


if __name__ == "__main__":
    main()
