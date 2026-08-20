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
import json
import os
import re
import shutil
import time
import uuid

import torch
import torch.utils.cpp_extension

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
# CUDA architecture detection and flags for Hopper (sm_90) and newer.

def _get_cuda_arch_flags():
    """Get CUDA architecture flags for compilation, supporting H200/Hopper (sm_90)."""
    arch_flags = []
    
    # Detect current GPU compute capability
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        cc = major * 10 + minor
        
        # Build gencode flags for detected architecture
        # Support architectures from Ampere (80) to Hopper (90) and beyond
        supported_archs = []
        
        # Add current GPU architecture
        if cc >= 70:  # Volta and newer
            supported_archs.append(cc)
        
        # Ensure we have at least Ampere support
        if 80 not in supported_archs and cc >= 80:
            supported_archs.append(80)
            
        # Add Hopper support for H200
        if 90 not in supported_archs and cc >= 90:
            supported_archs.append(90)
        
        # Generate flags
        for arch in sorted(set(supported_archs)):
            arch_flags.append(f'-gencode=arch=compute_{arch},code=sm_{arch}')
        
        # Add PTX for forward compatibility with the highest supported arch
        if supported_archs:
            max_arch = max(supported_archs)
            arch_flags.append(f'-gencode=arch=compute_{max_arch},code=compute_{max_arch}')
    
    return arch_flags

def _get_cuda_extra_cflags():
    """Get extra CUDA compilation flags optimized for modern architectures."""
    cflags = [
        '--use_fast_math',
        '-O3',
        '--expt-relaxed-constexpr',
        # Note: -fPIC is already added by PyTorch's cpp_extension loader
    ]
    
    # Add architecture-specific optimizations
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        
        # Hopper (sm_90) specific optimizations
        if major >= 9:
            cflags.extend([
                '-DHOPPER_ARCH',
                '--threads', '4',  # Parallel compilation
            ])
        # Ampere (sm_80) optimizations
        elif major >= 8:
            cflags.extend([
                '-DAMPERE_ARCH',
                '--threads', '4',
            ])
    
    return cflags

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

def _ensure_interpreter_bin_on_path():
    """Put the running interpreter's own ``bin/`` on ``PATH``.

    torch's C++ extension builder shells out to ``ninja`` found on ``PATH``.
    Calling this env's python by absolute path -- which every launcher that does
    not ``conda activate`` does, including the run-skill driver and the sh/
    scripts -- leaves the env's ``bin/`` off ``PATH``, so the build fails with
    "Ninja is required to load C++ extensions" even though ninja is installed
    right next to the interpreter.
    """
    import sys
    bindir = os.path.dirname(sys.executable)
    path = os.environ.get('PATH', '')
    if bindir and bindir not in path.split(os.pathsep):
        os.environ['PATH'] = bindir + os.pathsep + path if path else bindir
    return bindir

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

    _ensure_interpreter_bin_on_path()
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

    # Merge user-provided CUDA flags with architecture-specific flags
    extra_cuda_cflags = build_kwargs.pop('extra_cuda_cflags', [])
    if isinstance(extra_cuda_cflags, str):
        extra_cuda_cflags = [extra_cuda_cflags]
    else:
        extra_cuda_cflags = list(extra_cuda_cflags)
    
    # Add optimized flags for modern CUDA/architectures
    extra_cuda_cflags.extend(_get_cuda_extra_cflags())
    
    # Get architecture flags (gencode for Hopper/H200 support)
    arch_flags = _get_cuda_arch_flags()
    if arch_flags:
        extra_cuda_cflags.extend(arch_flags)
    
    # Restore extra_cuda_cflags to build_kwargs
    build_kwargs['extra_cuda_cflags'] = extra_cuda_cflags

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
            
            # Include CUDA flags in hash to rebuild when flags change
            hash_md5.update(str(extra_cuda_cflags).encode())

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
