#!/usr/bin/env python3
"""
Diagnostic script for GAN training performance issues.
Run this to identify potential bottlenecks and compatibility issues.
"""

import os
import subprocess

import torch


def print_section(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def check_cuda_environment():
    print_section("CUDA Environment")
    
    # PyTorch CUDA info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version (PyTorch): {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    - Compute capability: {props.major}.{props.minor}")
            print(f"    - Total memory: {props.total_memory / 1024**3:.1f} GB")
    
    # Check TORCH_CUDA_ARCH_LIST
    arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', 'Not set')
    print(f"\nTORCH_CUDA_ARCH_LIST: {arch_list}")
    
    # Recommended arch for H200 (Hopper)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"\nRecommended TORCH_CUDA_ARCH_LIST for your GPU: {props.major}.{props.minor}")

def check_gcc_version():
    print_section("GCC/Compiler Environment")
    
    try:
        result = subprocess.run(['gcc', '--version'], capture_output=True, text=True)
        print(f"GCC version:\n{result.stdout.split(chr(10))[0]}")
    except:
        print("Could not detect GCC version")
    
    # Check CUDA_HOME
    cuda_home = os.environ.get('CUDA_HOME', os.environ.get('CUDA_PATH', 'Not set'))
    print(f"\nCUDA_HOME: {cuda_home}")
    
    # Check nvcc
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'release' in line.lower():
                print(f"nvcc version: {line.strip()}")
    except:
        print("Could not detect nvcc version")

def check_nccl():
    print_section("NCCL Configuration")
    
    nccl_vars = ['NCCL_DEBUG', 'NCCL_DEBUG_SUBSYS', 'NCCL_P2P_DISABLE', 
                 'NCCL_SHM_DISABLE', 'NCCL_SOCKET_IFNAME', 'NCCL_IB_DISABLE']
    
    for var in nccl_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")

def check_custom_ops():
    print_section("Custom CUDA Operations Test")
    
    try:
        from torch_utils.ops import bias_act, upfirdn2d
        
        device = torch.device('cuda:0')
        
        # Test bias_act
        print("\nTesting bias_act...")
        x = torch.randn(4, 64, 32, 32, device=device)
        b = torch.randn(64, device=device)
        
        import time
        
        # Warmup
        for _ in range(10):
            bias_act.bias_act(x, b, act='lrelu')
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(100):
            bias_act.bias_act(x, b, act='lrelu')
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"  bias_act CUDA: {elapsed*10:.2f} ms per call")
        
        # Test with ref implementation
        start = time.time()
        for _ in range(100):
            bias_act.bias_act(x, b, act='lrelu', impl='ref')
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
            upfirdn2d.upfirdn2d(x, f, up=2)
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(100):
            upfirdn2d.upfirdn2d(x, f, up=2)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"  upfirdn2d CUDA: {elapsed*10:.2f} ms per call")
        
        start = time.time()
        for _ in range(100):
            upfirdn2d.upfirdn2d(x, f, up=2, impl='ref')
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

def check_pytorch_settings():
    print_section("PyTorch Performance Settings")
    
    print(f"torch.backends.cudnn.enabled: {torch.backends.cudnn.enabled}")
    print(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    print(f"torch.backends.cudnn.deterministic: {torch.backends.cudnn.deterministic}")
    print(f"torch.backends.cuda.matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"torch.backends.cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}")

def suggest_fixes():
    print_section("RECOMMENDED FIXES")
    
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    arch = f"{props.major}.{props.minor}" if props else "9.0"
    
    print("""
1. SET TORCH_CUDA_ARCH_LIST environment variable:
   For H200 (Hopper architecture), add to your script:
   
   export TORCH_CUDA_ARCH_LIST="{arch}"

2. USE AN OLDER GCC VERSION (Recommended):
   GCC 14.3 is too new. Install and use GCC 11 or 12:
   
   conda install -c conda-forge gcc=12 gxx=12
   # OR
   conda install -c conda-forge gcc=11 gxx=11

3. CLEAR CACHED CUDA KERNELS:
   rm -rf ~/.cache/torch_extensions/

4. SET NCCL ENVIRONMENT VARIABLES for multi-GPU:
   export NCCL_DEBUG=INFO
   export NCCL_P2P_LEVEL=NVL  # For NVLink systems like H200

5. TRY DISABLING CUSTOM CUDA OPS (for testing):
   In your training script, you can force PyTorch ops:
   - Edit torch_utils/ops/bias_act.py: change impl='cuda' to impl='ref'
   - Edit torch_utils/ops/upfirdn2d.py: change impl='cuda' to impl='ref'
   - Edit torch_utils/ops/filtered_lrelu.py: change impl='cuda' to impl='ref'
   
   If training is still slow with 'ref', the issue is elsewhere.
   If training speeds up with 'ref', the custom CUDA kernels are the problem.

6. CHECK FOR CUDA STREAM ISSUES:
   The warning about non-default CUDA streams suggests potential serialization.
   Try setting:
   
   export CUDA_LAUNCH_BLOCKING=0
""".format(arch=arch))

def main():
    print("GAN Training Performance Diagnostic Tool")
    print("========================================")
    
    check_cuda_environment()
    check_gcc_version()
    check_nccl()
    check_pytorch_settings()
    check_custom_ops()
    suggest_fixes()
    
    print("\n" + "="*60)
    print(" Diagnosis Complete")
    print("="*60)

if __name__ == "__main__":
    main()
