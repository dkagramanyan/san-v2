#!/usr/bin/env python3
import os, shutil, subprocess, sys

def run(cmd):
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print(f"\n$ {cmd}")
    if p.stdout: print(p.stdout.rstrip())
    if p.stderr: print(p.stderr.rstrip())
    return p.returncode

for c in [
    "nvidia-smi",
    "which nvcc",
    "nvcc --version",
    "which ninja",
    "g++ --version",
    r'conda list | grep -E "torch|cuda|cudnn"',
]:
    run(c)

print(f"\npython: {sys.executable}")
print(f"CONDA_PREFIX: {os.environ.get('CONDA_PREFIX')}")
print(f"nvcc: {shutil.which('nvcc')}")
print(f"ninja: {shutil.which('ninja')}")

import torch  # noqa: F401
from torch_utils import custom_ops
custom_ops.verbosity = "full"
from torch_utils.ops import upfirdn2d, bias_act, filtered_lrelu

print("Compiling upfirdn2d..."); upfirdn2d._init()
print("Compiling bias_act..."); bias_act._init()
print("Compiling filtered_lrelu..."); filtered_lrelu._init()
print("Pre-compilation complete!")
