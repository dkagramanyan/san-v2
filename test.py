#!/usr/bin/env python3
import os, shutil, subprocess, sys
import torch  # noqa: F401
from torch_utils import custom_ops
from torch_utils.ops import upfirdn2d, bias_act, filtered_lrelu


def check_custom_ops():
    print("Custom CUDA Operations Test")
    
    try:
        from torch_utils.ops import bias_act, filtered_lrelu, upfirdn2d
        
        device = torch.device('cuda:0')
        
        # Test bias_act
        print("\nTesting bias_act...")
        x = torch.randn(4, 64, 32, 32, device=device)
        b = torch.randn(64, device=device)
        
        import time
        
        # Warmup
        for _ in range(10):
            y = bias_act.bias_act(x, b, act='lrelu')
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(100):
            y = bias_act.bias_act(x, b, act='lrelu')
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"  bias_act CUDA: {elapsed*10:.2f} ms per call")
        
        # Test with ref implementation
        start = time.time()
        for _ in range(100):
            y = bias_act.bias_act(x, b, act='lrelu', impl='ref')
        torch.cuda.synchronize()
        elapsed_ref = time.time() - start
        print(f"  bias_act ref:  {elapsed_ref*10:.2f} ms per call")
        print(f"  CUDA speedup: {elapsed_ref/elapsed:.1f}x")
        
        if elapsed > elapsed_ref * 0.8:
            print("  WARNING: CUDA implementation is not faster than reference!")
            print("           This indicates custom CUDA kernels are not working properly.")
        
        # Test upfirdn2d
        print("\nTesting upfirdn2d...")
        f = upfirdn2d.setup_filter([1, 3, 3, 1], device=device)
        
        # Warmup
        for _ in range(10):
            y = upfirdn2d.upfirdn2d(x, f, up=2)
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(100):
            y = upfirdn2d.upfirdn2d(x, f, up=2)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"  upfirdn2d CUDA: {elapsed*10:.2f} ms per call")
        
        start = time.time()
        for _ in range(100):
            y = upfirdn2d.upfirdn2d(x, f, up=2, impl='ref')
        torch.cuda.synchronize()
        elapsed_ref = time.time() - start
        print(f"  upfirdn2d ref:  {elapsed_ref*10:.2f} ms per call")
        print(f"  CUDA speedup: {elapsed_ref/elapsed:.1f}x")
        
        if elapsed > elapsed_ref * 0.8:
            print("  WARNING: CUDA implementation is not faster than reference!")
        
    except Exception as e:
        print(f"Error testing custom ops: {e}")
        import traceback
        traceback.print_exc()


def run(cmd):
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print(f"\n$ {cmd}")
    if p.stdout: print(p.stdout.rstrip())
    if p.stderr: print(p.stderr.rstrip())
    return p.returncode

custom_ops.verbosity = "full"

for c in [
    'rm -rf ~/.cache/torch_extensions',
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


print("torch cuda:", torch.version.cuda)
print("is available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))

print("Compiling upfirdn2d..."); upfirdn2d._init()
print("Compiling bias_act..."); bias_act._init()
print("Compiling filtered_lrelu..."); filtered_lrelu._init()
print("Pre-compilation complete!")



check_custom_ops()