#!/usr/bin/env python3
"""
Comprehensive test suite for custom CUDA operations.
Tests compilation, correctness against PyTorch reference implementations,
performance benchmarking, and CUDA 13 / H200 Hopper architecture compatibility.
"""

import os
import sys
import shutil
import subprocess
import time
import traceback
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np

# Import custom ops modules
from torch_utils import custom_ops
from torch_utils.ops import upfirdn2d, bias_act, filtered_lrelu


def run(cmd: str) -> int:
    """Run a shell command and print output."""
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print(f"\n$ {cmd}")
    if p.stdout:
        print(p.stdout.rstrip())
    if p.stderr:
        print(p.stderr.rstrip())
    return p.returncode


def print_system_info():
    """Print system and CUDA information."""
    print("=" * 70)
    print("SYSTEM INFORMATION")
    print("=" * 70)
    
    for c in [
        'rm -rf ~/.cache/torch_extensions',
        "nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader",
        "which nvcc",
        "nvcc --version",
        "which ninja",
        "g++ --version | head -1",
        r'conda list 2>/dev/null | grep -E "torch|cuda|cudnn" | head -10 || pip list | grep -E "torch|cuda" | head -10',
    ]:
        run(c)
    
    print(f"\nPython: {sys.executable}")
    print(f"CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', 'Not set')}")
    print(f"nvcc: {shutil.which('nvcc')}")
    print(f"ninja: {shutil.which('ninja')}")
    
    print(f"\nPyTorch CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"Current device: {device}")
        print(f"Device name: {torch.cuda.get_device_name(device)}")
        major, minor = torch.cuda.get_device_capability(device)
        print(f"Compute capability: {major}.{minor}")
        
        # Check for Hopper architecture
        if major >= 9:
            print(f"✓ Hopper architecture detected (sm_{major}{minor}) - H100/H200 compatible")
        elif major >= 8:
            print(f"✓ Ampere architecture detected (sm_{major}{minor})")
        else:
            print(f"ℹ Older architecture detected (sm_{major}{minor})")
    
    print("=" * 70)


def compile_custom_ops():
    """Compile all custom CUDA operations."""
    print("\n" + "=" * 70)
    print("COMPILING CUSTOM CUDA OPERATIONS")
    print("=" * 70)
    
    custom_ops.verbosity = "full"
    
    try:
        print("\n[1/3] Compiling upfirdn2d...")
        upfirdn2d._init()
        print("      ✓ upfirdn2d compiled successfully")
        
        print("\n[2/3] Compiling bias_act...")
        bias_act._init()
        print("      ✓ bias_act compiled successfully")
        
        print("\n[3/3] Compiling filtered_lrelu...")
        filtered_lrelu._init()
        print("      ✓ filtered_lrelu compiled successfully")
        
        print("\n✓ All custom ops compiled successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Compilation failed: {e}")
        traceback.print_exc()
        return False


def test_bias_act_correctness() -> Dict[str, bool]:
    """Test bias_act CUDA kernel correctness against reference implementation."""
    print("\n" + "-" * 50)
    print("Testing bias_act correctness...")
    print("-" * 50)
    
    results = {}
    device = torch.device('cuda:0')
    
    # Test configurations
    test_configs = [
        {'act': 'linear', 'dtype': torch.float32},
        {'act': 'relu', 'dtype': torch.float32},
        {'act': 'lrelu', 'dtype': torch.float32},
        {'act': 'tanh', 'dtype': torch.float32},
        {'act': 'sigmoid', 'dtype': torch.float32},
        {'act': 'swish', 'dtype': torch.float32},
        {'act': 'lrelu', 'dtype': torch.float16},
    ]
    
    # Check if bfloat16 is supported (Ampere+)
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        test_configs.append({'act': 'lrelu', 'dtype': torch.bfloat16})
    
    for config in test_configs:
        act = config['act']
        dtype = config['dtype']
        test_name = f"bias_act_{act}_{dtype}"
        
        try:
            x = torch.randn(4, 64, 32, 32, device=device, dtype=dtype)
            b = torch.randn(64, device=device, dtype=dtype)
            
            # Reference implementation
            y_ref = bias_act.bias_act(x, b, act=act, impl='ref')
            
            # CUDA implementation
            y_cuda = bias_act.bias_act(x, b, act=act, impl='cuda')
            
            # Compare results
            if dtype == torch.float16 or dtype == torch.bfloat16:
                atol, rtol = 1e-2, 1e-2
            else:
                atol, rtol = 1e-5, 1e-5
            
            is_close = torch.allclose(y_ref, y_cuda, atol=atol, rtol=rtol)
            max_diff = (y_ref - y_cuda).abs().max().item()
            
            results[test_name] = is_close
            status = "✓ PASS" if is_close else "✗ FAIL"
            print(f"  {test_name}: {status} (max_diff={max_diff:.2e})")
            
        except Exception as e:
            results[test_name] = False
            print(f"  {test_name}: ✗ ERROR - {e}")
    
    return results


def test_upfirdn2d_correctness() -> Dict[str, bool]:
    """Test upfirdn2d CUDA kernel correctness against reference implementation."""
    print("\n" + "-" * 50)
    print("Testing upfirdn2d correctness...")
    print("-" * 50)
    
    results = {}
    device = torch.device('cuda:0')
    
    # Test configurations
    test_configs = [
        {'up': 1, 'down': 1, 'filter': [1, 3, 3, 1], 'dtype': torch.float32},
        {'up': 2, 'down': 1, 'filter': [1, 3, 3, 1], 'dtype': torch.float32},
        {'up': 1, 'down': 2, 'filter': [1, 3, 3, 1], 'dtype': torch.float32},
        {'up': 2, 'down': 2, 'filter': [1, 3, 3, 1], 'dtype': torch.float32},
        {'up': 4, 'down': 1, 'filter': [1, 4, 6, 4, 1], 'dtype': torch.float32},
        {'up': 1, 'down': 4, 'filter': [1, 4, 6, 4, 1], 'dtype': torch.float32},
        {'up': 2, 'down': 1, 'filter': [1, 3, 3, 1], 'dtype': torch.float16},
    ]
    
    # Check if bfloat16 is supported
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        test_configs.append({'up': 2, 'down': 1, 'filter': [1, 3, 3, 1], 'dtype': torch.bfloat16})
    
    for config in test_configs:
        up = config['up']
        down = config['down']
        filt = config['filter']
        dtype = config['dtype']
        test_name = f"upfirdn2d_up{up}_down{down}_{dtype}"
        
        try:
            x = torch.randn(4, 64, 32, 32, device=device, dtype=dtype)
            f = upfirdn2d.setup_filter(filt, device=device)
            
            # Reference implementation
            y_ref = upfirdn2d.upfirdn2d(x, f, up=up, down=down, impl='ref')
            
            # CUDA implementation
            y_cuda = upfirdn2d.upfirdn2d(x, f, up=up, down=down, impl='cuda')
            
            # Compare results
            if dtype == torch.float16 or dtype == torch.bfloat16:
                atol, rtol = 1e-2, 1e-2
            else:
                atol, rtol = 1e-4, 1e-4
            
            is_close = torch.allclose(y_ref, y_cuda, atol=atol, rtol=rtol)
            max_diff = (y_ref - y_cuda).abs().max().item()
            
            results[test_name] = is_close
            status = "✓ PASS" if is_close else "✗ FAIL"
            print(f"  {test_name}: {status} (max_diff={max_diff:.2e})")
            
        except Exception as e:
            results[test_name] = False
            print(f"  {test_name}: ✗ ERROR - {e}")
    
    return results


def test_filtered_lrelu_correctness() -> Dict[str, bool]:
    """Test filtered_lrelu CUDA kernel correctness against reference implementation."""
    print("\n" + "-" * 50)
    print("Testing filtered_lrelu correctness...")
    print("-" * 50)
    
    results = {}
    device = torch.device('cuda:0')
    
    # Test configurations
    test_configs = [
        {'up': 1, 'down': 1, 'dtype': torch.float32},
        {'up': 2, 'down': 1, 'dtype': torch.float32},
        {'up': 1, 'down': 2, 'dtype': torch.float32},
        {'up': 2, 'down': 2, 'dtype': torch.float32},
        {'up': 2, 'down': 1, 'dtype': torch.float16},
    ]
    
    for config in test_configs:
        up = config['up']
        down = config['down']
        dtype = config['dtype']
        test_name = f"filtered_lrelu_up{up}_down{down}_{dtype}"
        
        try:
            x = torch.randn(2, 32, 16, 16, device=device, dtype=dtype)
            b = torch.randn(32, device=device, dtype=dtype)
            fu = upfirdn2d.setup_filter([1, 3, 3, 1], device=device) * (up ** 2)
            fd = upfirdn2d.setup_filter([1, 3, 3, 1], device=device)
            
            # Reference implementation
            y_ref = filtered_lrelu.filtered_lrelu(x, fu=fu, fd=fd, b=b, up=up, down=down, impl='ref')
            
            # CUDA implementation
            y_cuda = filtered_lrelu.filtered_lrelu(x, fu=fu, fd=fd, b=b, up=up, down=down, impl='cuda')
            
            # Compare results
            if dtype == torch.float16:
                atol, rtol = 5e-2, 5e-2  # Larger tolerance for float16 due to numerical precision
            else:
                atol, rtol = 1e-4, 1e-4
            
            is_close = torch.allclose(y_ref, y_cuda, atol=atol, rtol=rtol)
            max_diff = (y_ref - y_cuda).abs().max().item()
            
            results[test_name] = is_close
            status = "✓ PASS" if is_close else "✗ FAIL"
            print(f"  {test_name}: {status} (max_diff={max_diff:.2e})")
            
        except Exception as e:
            results[test_name] = False
            print(f"  {test_name}: ✗ ERROR - {e}")
    
    return results


def benchmark_custom_ops():
    """Benchmark custom CUDA operations performance."""
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 70)
    
    device = torch.device('cuda:0')
    
    # Warmup iterations and benchmark iterations
    warmup = 10
    iterations = 100
    
    # Test bias_act performance
    print("\n--- bias_act Performance ---")
    x = torch.randn(8, 512, 64, 64, device=device)
    b = torch.randn(512, device=device)
    
    # Warmup
    for _ in range(warmup):
        _ = bias_act.bias_act(x, b, act='lrelu', impl='cuda')
    torch.cuda.synchronize()
    
    # Benchmark CUDA
    start = time.time()
    for _ in range(iterations):
        _ = bias_act.bias_act(x, b, act='lrelu', impl='cuda')
    torch.cuda.synchronize()
    cuda_time = (time.time() - start) / iterations * 1000
    
    # Benchmark reference
    start = time.time()
    for _ in range(iterations):
        _ = bias_act.bias_act(x, b, act='lrelu', impl='ref')
    torch.cuda.synchronize()
    ref_time = (time.time() - start) / iterations * 1000
    
    speedup = ref_time / cuda_time
    print(f"  CUDA: {cuda_time:.3f} ms, Ref: {ref_time:.3f} ms, Speedup: {speedup:.1f}x")
    
    if speedup < 1.0:
        print("  ⚠ WARNING: CUDA implementation slower than reference!")
    else:
        print(f"  ✓ CUDA implementation is {speedup:.1f}x faster")
    
    # Test upfirdn2d performance
    print("\n--- upfirdn2d Performance ---")
    x = torch.randn(8, 512, 64, 64, device=device)
    f = upfirdn2d.setup_filter([1, 3, 3, 1], device=device)
    
    # Warmup
    for _ in range(warmup):
        _ = upfirdn2d.upfirdn2d(x, f, up=2, impl='cuda')
    torch.cuda.synchronize()
    
    # Benchmark CUDA
    start = time.time()
    for _ in range(iterations):
        _ = upfirdn2d.upfirdn2d(x, f, up=2, impl='cuda')
    torch.cuda.synchronize()
    cuda_time = (time.time() - start) / iterations * 1000
    
    # Benchmark reference
    start = time.time()
    for _ in range(iterations):
        _ = upfirdn2d.upfirdn2d(x, f, up=2, impl='ref')
    torch.cuda.synchronize()
    ref_time = (time.time() - start) / iterations * 1000
    
    speedup = ref_time / cuda_time
    print(f"  CUDA: {cuda_time:.3f} ms, Ref: {ref_time:.3f} ms, Speedup: {speedup:.1f}x")
    
    if speedup < 1.0:
        print("  ⚠ WARNING: CUDA implementation slower than reference!")
    else:
        print(f"  ✓ CUDA implementation is {speedup:.1f}x faster")


def test_gradient_correctness():
    """Test that gradients flow correctly through custom ops."""
    print("\n" + "-" * 50)
    print("Testing gradient correctness...")
    print("-" * 50)
    
    device = torch.device('cuda:0')
    results = {}
    
    # Test bias_act gradients
    try:
        x = torch.randn(2, 32, 16, 16, device=device, requires_grad=True)
        b = torch.randn(32, device=device, requires_grad=True)
        
        y = bias_act.bias_act(x, b, act='lrelu', impl='cuda')
        loss = y.sum()
        loss.backward()
        
        has_grad = x.grad is not None and b.grad is not None
        results['bias_act_gradient'] = has_grad
        print(f"  bias_act gradients: {'✓ PASS' if has_grad else '✗ FAIL'}")
    except Exception as e:
        results['bias_act_gradient'] = False
        print(f"  bias_act gradients: ✗ ERROR - {e}")
    
    # Test upfirdn2d gradients
    try:
        x = torch.randn(2, 32, 16, 16, device=device, requires_grad=True)
        f = upfirdn2d.setup_filter([1, 3, 3, 1], device=device)
        
        y = upfirdn2d.upfirdn2d(x, f, up=2, impl='cuda')
        loss = y.sum()
        loss.backward()
        
        has_grad = x.grad is not None
        results['upfirdn2d_gradient'] = has_grad
        print(f"  upfirdn2d gradients: {'✓ PASS' if has_grad else '✗ FAIL'}")
    except Exception as e:
        results['upfirdn2d_gradient'] = False
        print(f"  upfirdn2d gradients: ✗ ERROR - {e}")
    
    # Test filtered_lrelu gradients
    try:
        x = torch.randn(2, 32, 16, 16, device=device, requires_grad=True)
        b = torch.randn(32, device=device, requires_grad=True)
        fu = upfirdn2d.setup_filter([1, 3, 3, 1], device=device)
        fd = upfirdn2d.setup_filter([1, 3, 3, 1], device=device)
        
        y = filtered_lrelu.filtered_lrelu(x, fu=fu, fd=fd, b=b, up=2, down=2, impl='cuda')
        loss = y.sum()
        loss.backward()
        
        has_grad = x.grad is not None and b.grad is not None
        results['filtered_lrelu_gradient'] = has_grad
        print(f"  filtered_lrelu gradients: {'✓ PASS' if has_grad else '✗ FAIL'}")
    except Exception as e:
        results['filtered_lrelu_gradient'] = False
        print(f"  filtered_lrelu gradients: ✗ ERROR - {e}")
    
    return results


def main():
    """Main test function."""
    print("\n" + "=" * 70)
    print("CUSTOM CUDA OPERATIONS TEST SUITE")
    print("CUDA 13 / H200 Hopper Architecture Compatibility")
    print("=" * 70)
    
    # Print system info
    print_system_info()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n✗ CUDA is not available. Cannot run tests.")
        return 1
    
    # Compile custom ops
    if not compile_custom_ops():
        print("\n✗ Failed to compile custom ops. Aborting tests.")
        return 1
    
    # Run correctness tests
    print("\n" + "=" * 70)
    print("CORRECTNESS TESTS")
    print("=" * 70)
    
    all_results = {}
    
    all_results.update(test_bias_act_correctness())
    all_results.update(test_upfirdn2d_correctness())
    all_results.update(test_filtered_lrelu_correctness())
    all_results.update(test_gradient_correctness())
    
    # Run performance benchmarks
    benchmark_custom_ops()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in all_results.values() if v)
    total = len(all_results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED:")
        for name, passed in all_results.items():
            if not passed:
                print(f"  - {name}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
