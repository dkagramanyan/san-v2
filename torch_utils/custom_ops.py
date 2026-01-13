# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import glob
import hashlib
import importlib
import os
import re
import shutil
import uuid
import time
import json

import torch
import torch.utils.cpp_extension
from torch.utils.file_baton import FileBaton

#----------------------------------------------------------------------------
# Global options.

verbosity = 'brief' # Verbosity level: 'none', 'brief', 'full'

#----------------------------------------------------------------------------
# Debug logging for CUDA kernel diagnostics

# #region agent log
_DEBUG_LOG_PATH = "/home/dgkagramanyan/.cursor/debug.log"

def _debug_log(location, message, data=None, hypothesis_id=None):
    """Write debug info to NDJSON log file for kernel compilation diagnostics."""
    try:
        entry = {
            "timestamp": time.time() * 1000,
            "location": location,
            "message": message,
            "data": data or {},
            "sessionId": "cuda-kernel-debug",
            "hypothesisId": hypothesis_id or "general"
        }
        with open(_DEBUG_LOG_PATH, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    except Exception:
        pass  # Fail silently to not disrupt training
# #endregion

#----------------------------------------------------------------------------
# Get CUDA architecture flags for H200/Hopper (sm_90) and A100 (sm_80)

def _get_cuda_arch_flags():
    """Return CUDA architecture flags targeting Hopper (sm_90) and Ampere (sm_80)."""
    try:
        major, minor = torch.cuda.get_device_capability()
        device_name = torch.cuda.get_device_name()

        # Get additional Hopper-specific info (fixed property name for PyTorch 2.9+)
        props = torch.cuda.get_device_properties(0)
        shared_mem = props.shared_memory_per_block
        multiprocessor_count = props.multi_processor_count
        total_memory = props.total_memory
    except Exception:
        major, minor = 9, 0  # Default to Hopper
        device_name = "unknown"
        shared_mem = 0
        multiprocessor_count = 0
        total_memory = 0
    
    # #region agent log
    _debug_log("custom_ops.py:_get_cuda_arch_flags", "Detected GPU capability", {
        "major": major,
        "minor": minor,
        "device_name": device_name,
        "compute_capability": f"sm_{major}{minor}",
        "shared_mem_per_block_bytes": shared_mem,
        "shared_mem_per_block_kb": shared_mem / 1024,
        "multiprocessor_count": multiprocessor_count,
        "total_memory_gb": total_memory / (1024**3),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__
    }, "A")
    # #endregion
    
    # Always include Hopper (sm_90) for H200 and Ampere (sm_80) for A100
    arch_flags = [
        '-gencode=arch=compute_80,code=sm_80',   # A100 (Ampere)
        '-gencode=arch=compute_90,code=sm_90',   # H100/H200 (Hopper)
        '-gencode=arch=compute_90,code=compute_90'  # PTX for forward compatibility
    ]
    
    # Add current device's arch explicitly if different
    current_arch = f'-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}'
    if current_arch not in arch_flags:
        arch_flags.insert(0, current_arch)
    
    # #region agent log
    _debug_log("custom_ops.py:_get_cuda_arch_flags", "Architecture flags selected", {
        "arch_flags": arch_flags
    }, "A")
    # #endregion
    
    return arch_flags

#----------------------------------------------------------------------------
# Internal helper funcs.

def _find_compiler_bindir():
    patterns = [
        'C:/Program Files (x86)/Microsoft Visual Studio/*/Professional/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files (x86)/Microsoft Visual Studio/*/BuildTools/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files (x86)/Microsoft Visual Studio/*/Community/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files (x86)/Microsoft Visual Studio */vc/bin',
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if len(matches):
            return matches[-1]
    return None

#----------------------------------------------------------------------------

def _get_mangled_gpu_name():
    name = torch.cuda.get_device_name().lower()
    out = []
    for c in name:
        if re.match('[a-z0-9_-]+', c):
            out.append(c)
        else:
            out.append('-')
    return ''.join(out)

#----------------------------------------------------------------------------
# Main entry point for compiling and loading C++/CUDA plugins.

_cached_plugins = dict()

def get_plugin(module_name, sources, headers=None, source_dir=None, **build_kwargs):
    import sys
    assert verbosity in ['none', 'brief', 'full']
    if headers is None:
        headers = []
    if source_dir is not None:
        sources = [os.path.join(source_dir, fname) for fname in sources]
        headers = [os.path.join(source_dir, fname) for fname in headers]

    if module_name in _cached_plugins:
        # #region agent log
        _debug_log(f"custom_ops.py:get_plugin:{module_name}", "Plugin found in memory cache", {
            "module_name": module_name,
            "from_cache": True
        }, "B")
        # #endregion
        return _cached_plugins[module_name]

    start_time = time.time()
    
    # #region agent log
    _debug_log(f"custom_ops.py:get_plugin:{module_name}", "Starting plugin setup", {
        "module_name": module_name,
        "sources": sources,
        "build_kwargs_keys": list(build_kwargs.keys())
    }, "A")
    # #endregion

    if verbosity == 'full':
        print(f'Setting up PyTorch plugin "{module_name}"...')
    elif verbosity == 'brief':
        print(f'Setting up PyTorch plugin "{module_name}"... ', end='', flush=True)
    verbose_build = (verbosity == 'full')

    build_dir = None
    
    # Inject Hopper architecture flags into extra_cuda_cflags
    cuda_arch_flags = _get_cuda_arch_flags()
    if 'extra_cuda_cflags' in build_kwargs:
        original_flags = build_kwargs['extra_cuda_cflags']
        build_kwargs['extra_cuda_cflags'] = cuda_arch_flags + list(original_flags)
    else:
        build_kwargs['extra_cuda_cflags'] = cuda_arch_flags
    
    # #region agent log
    _debug_log(f"custom_ops.py:get_plugin:{module_name}", "CUDA flags configured", {
        "extra_cuda_cflags": build_kwargs['extra_cuda_cflags']
    }, "A")
    # #endregion

    try:
        if os.name == 'nt' and os.system("where cl.exe >nul 2>nul") != 0:
            compiler_bindir = _find_compiler_bindir()
            if compiler_bindir is None:
                raise RuntimeError(f'Could not find MSVC/GCC/CLANG installation on this computer. Check _find_compiler_bindir() in "{__file__}".')
            os.environ['PATH'] += ';' + compiler_bindir

        all_source_files = sorted(sources + headers)
        all_source_dirs = set(os.path.dirname(fname) for fname in all_source_files)

        if len(all_source_dirs) == 1:
            hash_md5 = hashlib.md5()
            for src in all_source_files:
                with open(src, 'rb') as f:
                    hash_md5.update(f.read())
            
            # Include architecture flags in hash to invalidate cache when flags change
            arch_flag_str = '|'.join(cuda_arch_flags)
            hash_md5.update(arch_flag_str.encode('utf-8'))

            source_digest = hash_md5.hexdigest()
            build_top_dir = torch.utils.cpp_extension._get_build_directory(module_name, verbose=verbose_build)  # pylint: disable=protected-access
            cached_build_dir = os.path.join(build_top_dir, f'{source_digest}-{_get_mangled_gpu_name()}')
            build_dir = cached_build_dir

            cache_exists = os.path.isdir(cached_build_dir)
            
            # #region agent log
            _debug_log(f"custom_ops.py:get_plugin:{module_name}", "Build directory check", {
                "cached_build_dir": cached_build_dir,
                "cache_exists": cache_exists,
                "source_digest": source_digest
            }, "B")
            # #endregion

            if not cache_exists:
                # #region agent log
                _debug_log(f"custom_ops.py:get_plugin:{module_name}", "Cache miss - compiling fresh", {
                    "reason": "directory_not_found"
                }, "B")
                # #endregion
                
                tmpdir = f'{build_top_dir}/srctmp-{uuid.uuid4().hex}'
                os.makedirs(tmpdir)
                for src in all_source_files:
                    shutil.copyfile(src, os.path.join(tmpdir, os.path.basename(src)))
                try:
                    os.replace(tmpdir, cached_build_dir)
                except OSError:
                    shutil.rmtree(tmpdir)
                    if not os.path.isdir(cached_build_dir):
                        raise

            cached_sources = [os.path.join(cached_build_dir, os.path.basename(fname)) for fname in sources]
            
            compile_start = time.time()
            torch.utils.cpp_extension.load(
                name=module_name,
                build_directory=cached_build_dir,
                verbose=verbose_build,
                sources=cached_sources,
                **build_kwargs,
            )
            compile_time = time.time() - compile_start
            
            # #region agent log
            _debug_log(f"custom_ops.py:get_plugin:{module_name}", "Plugin compilation/load complete", {
                "compile_time_sec": compile_time,
                "was_cached": cache_exists
            }, "A")
            # #endregion
        else:
            build_dir = torch.utils.cpp_extension._get_build_directory(module_name, verbose=verbose_build)  # pylint: disable=protected-access
            
            compile_start = time.time()
            torch.utils.cpp_extension.load(
                name=module_name,
                verbose=verbose_build,
                sources=sources,
                **build_kwargs,
            )
            compile_time = time.time() - compile_start
            
            # #region agent log
            _debug_log(f"custom_ops.py:get_plugin:{module_name}", "Plugin load complete (multi-dir)", {
                "compile_time_sec": compile_time
            }, "A")
            # #endregion

        # Ensure the directory containing *.so is importable
        if build_dir and os.path.isdir(build_dir) and build_dir not in sys.path:
            sys.path.insert(0, build_dir)

        module = importlib.import_module(module_name)

        total_time = time.time() - start_time
        
        # #region agent log
        _debug_log(f"custom_ops.py:get_plugin:{module_name}", "Plugin setup complete", {
            "total_time_sec": total_time,
            "build_dir": build_dir,
            "module_loaded": module_name
        }, "A")
        # #endregion

    except Exception as e:
        # #region agent log
        _debug_log(f"custom_ops.py:get_plugin:{module_name}", "Plugin setup FAILED", {
            "error": str(e),
            "error_type": type(e).__name__
        }, "A")
        # #endregion
        
        if verbosity == 'brief':
            print('Failed!')
        raise

    if verbosity == 'full':
        print(f'Done setting up PyTorch plugin "{module_name}".')
    elif verbosity == 'brief':
        print('Done.')

    _cached_plugins[module_name] = module
    return module
