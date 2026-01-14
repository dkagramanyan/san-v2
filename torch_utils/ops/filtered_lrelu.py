# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np
import torch
import warnings
import time
import json

from .. import custom_ops
from .. import misc
from . import upfirdn2d
from . import bias_act

#----------------------------------------------------------------------------

# #region agent log
_DEBUG_LOG_PATH = "/home/dgkagramanyan/.cursor/debug.log"
_kernel_call_count = 0
_kernel_config_times = {}

def _debug_log(location, message, data=None, hypothesis_id=None):
    """Write debug info to NDJSON log file for kernel execution diagnostics."""
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
        pass
# #endregion

_plugin = None

def _init():
    global _plugin
    if _plugin is None:
        # #region agent log
        init_start = time.time()
        _debug_log("filtered_lrelu.py:_init", "Initializing filtered_lrelu plugin", {}, "C")
        # #endregion
        
        _plugin = custom_ops.get_plugin(
            module_name='filtered_lrelu_plugin',
            sources=['filtered_lrelu.cpp', 'filtered_lrelu_wr.cu', 'filtered_lrelu_rd.cu', 'filtered_lrelu_ns.cu'],
            headers=['filtered_lrelu.h', 'filtered_lrelu.cu'],
            source_dir=os.path.dirname(__file__),
            extra_cuda_cflags=['--use_fast_math'],
        )
        
        # #region agent log
        _debug_log("filtered_lrelu.py:_init", "Plugin initialized", {
            "init_time_sec": time.time() - init_start
        }, "C")
        # #endregion
    return True

def _get_filter_size(f):
    if f is None:
        return 1, 1
    assert isinstance(f, torch.Tensor)
    assert 1 <= f.ndim <= 2
    return f.shape[-1], f.shape[0] # width, height

def _parse_padding(padding):
    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, (int, np.integer)) for x in padding)
    padding = [int(x) for x in padding]
    if len(padding) == 2:
        px, py = padding
        padding = [px, px, py, py]
    px0, px1, py0, py1 = padding
    return px0, px1, py0, py1

#----------------------------------------------------------------------------

def filtered_lrelu(x, fu=None, fd=None, b=None, up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False, impl='cuda'):
    r"""Filtered leaky ReLU for a batch of 2D images.

    Performs the following sequence of operations for each channel:

    1. Add channel-specific bias if provided (`b`).

    2. Upsample the image by inserting N-1 zeros after each pixel (`up`).

    3. Pad the image with the specified number of zeros on each side (`padding`).
       Negative padding corresponds to cropping the image.

    4. Convolve the image with the specified upsampling FIR filter (`fu`), shrinking it
       so that the footprint of all output pixels lies within the input image.

    5. Multiply each value by the provided gain factor (`gain`).

    6. Apply leaky ReLU activation function to each value.

    7. Clamp each value between -clamp and +clamp, if `clamp` parameter is provided.

    8. Convolve the image with the specified downsampling FIR filter (`fd`), shrinking
       it so that the footprint of all output pixels lies within the input image.

    9. Downsample the image by keeping every Nth pixel (`down`).

    The fused op is considerably more efficient than performing the same calculation
    using standard PyTorch ops. It supports gradients of arbitrary order.

    Args:
        x:           Float32/float16/float64 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        fu:          Float32 upsampling FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        fd:          Float32 downsampling FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        b:           Bias vector, or `None` to disable. Must be a 1D tensor of the same type
                     as `x`. The length of vector must must match the channel dimension of `x`.
        up:          Integer upsampling factor (default: 1).
        down:        Integer downsampling factor. (default: 1).
        padding:     Padding with respect to the upsampled image. Can be a single number
                     or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        gain:        Overall scaling factor for signal magnitude (default: sqrt(2)).
        slope:       Slope on the negative side of leaky ReLU (default: 0.2).
        clamp:       Maximum magnitude for leaky ReLU output (default: None).
        flip_filter: False = convolution, True = correlation (default: False).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    assert isinstance(x, torch.Tensor)
    assert impl in ['ref', 'cuda']
    if impl == 'cuda' and x.device.type == 'cuda' and _init():
        return _filtered_lrelu_cuda(up=up, down=down, padding=padding, gain=gain, slope=slope, clamp=clamp, flip_filter=flip_filter).apply(x, fu, fd, b, None, 0, 0)
    return _filtered_lrelu_ref(x, fu=fu, fd=fd, b=b, up=up, down=down, padding=padding, gain=gain, slope=slope, clamp=clamp, flip_filter=flip_filter)

#----------------------------------------------------------------------------

@misc.profiled_function
def _filtered_lrelu_ref(x, fu=None, fd=None, b=None, up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False):
    """Slow and memory-inefficient reference implementation of `filtered_lrelu()` using
    existing `upfirdn2n()` and `bias_act()` ops.
    """
    assert isinstance(x, torch.Tensor) and x.ndim == 4
    fu_w, fu_h = _get_filter_size(fu)
    fd_w, fd_h = _get_filter_size(fd)
    if b is not None:
        assert isinstance(b, torch.Tensor) and b.dtype == x.dtype
        misc.assert_shape(b, [x.shape[1]])
    assert isinstance(up, int) and up >= 1
    assert isinstance(down, int) and down >= 1
    px0, px1, py0, py1 = _parse_padding(padding)
    assert gain == float(gain) and gain > 0
    assert slope == float(slope) and slope >= 0
    assert clamp is None or (clamp == float(clamp) and clamp >= 0)

    # Calculate output size.
    batch_size, channels, in_h, in_w = x.shape
    in_dtype = x.dtype
    out_w = (in_w * up + (px0 + px1) - (fu_w - 1) - (fd_w - 1) + (down - 1)) // down
    out_h = (in_h * up + (py0 + py1) - (fu_h - 1) - (fd_h - 1) + (down - 1)) // down

    # Compute using existing ops.
    x = bias_act.bias_act(x=x, b=b) # Apply bias.
    x = upfirdn2d.upfirdn2d(x=x, f=fu, up=up, padding=[px0, px1, py0, py1], gain=up**2, flip_filter=flip_filter) # Upsample.
    x = bias_act.bias_act(x=x, act='lrelu', alpha=slope, gain=gain, clamp=clamp) # Bias, leaky ReLU, clamp.
    x = upfirdn2d.upfirdn2d(x=x, f=fd, down=down, flip_filter=flip_filter) # Downsample.

    # Check output shape & dtype.
    misc.assert_shape(x, [batch_size, channels, out_h, out_w])
    assert x.dtype == in_dtype
    return x

#----------------------------------------------------------------------------

_filtered_lrelu_cuda_cache = dict()

def _filtered_lrelu_cuda(up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False):
    """Fast CUDA implementation of `filtered_lrelu()` using custom ops.
    """
    assert isinstance(up, int) and up >= 1
    assert isinstance(down, int) and down >= 1
    px0, px1, py0, py1 = _parse_padding(padding)
    assert gain == float(gain) and gain > 0
    gain = float(gain)
    assert slope == float(slope) and slope >= 0
    slope = float(slope)
    assert clamp is None or (clamp == float(clamp) and clamp >= 0)
    clamp = float(clamp if clamp is not None else 'inf')

    # Lookup from cache.
    key = (up, down, px0, px1, py0, py1, gain, slope, clamp, flip_filter)
    if key in _filtered_lrelu_cuda_cache:
        return _filtered_lrelu_cuda_cache[key]
    
    # #region agent log
    _debug_log("filtered_lrelu.py:_filtered_lrelu_cuda", "Creating new kernel config class", {
        "key": str(key),
        "up": up,
        "down": down,
        "padding": [px0, px1, py0, py1]
    }, "C")
    # #endregion

    # Forward op.
    class FilteredLReluCuda(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, fu, fd, b, si, sx, sy): # pylint: disable=arguments-differ
            global _kernel_call_count, _kernel_config_times
            assert isinstance(x, torch.Tensor) and x.ndim == 4
            
            # #region agent log
            _kernel_call_count += 1
            config_key = f"{x.shape}_{up}_{down}"
            kernel_start = time.time()
            _timings = {}
            # #endregion

            # Replace empty up/downsample kernels with full 1x1 kernels (faster than separable).
            if fu is None:
                fu = torch.ones([1, 1], dtype=torch.float32, device=x.device)
            if fd is None:
                fd = torch.ones([1, 1], dtype=torch.float32, device=x.device)
            assert 1 <= fu.ndim <= 2
            assert 1 <= fd.ndim <= 2

            # Replace separable 1x1 kernels with full 1x1 kernels when scale factor is 1.
            if up == 1 and fu.ndim == 1 and fu.shape[0] == 1:
                fu = fu.square()[None]
            if down == 1 and fd.ndim == 1 and fd.shape[0] == 1:
                fd = fd.square()[None]

            # Missing sign input tensor.
            if si is None:
                si = torch.empty([0])

            # Missing bias tensor.
            if b is None:
                b = torch.zeros([x.shape[1]], dtype=x.dtype, device=x.device)

            # Construct internal sign tensor only if gradients are needed.
            write_signs = (si.numel() == 0) and (x.requires_grad or b.requires_grad)

            # #region agent log
            torch.cuda.synchronize()
            _t0_contig = time.time()
            # #endregion
            
            # Warn if input storage strides are not in decreasing order due to e.g. channels-last layout.
            x = x.contiguous()
            strides = [x.stride(i) for i in range(x.ndim) if x.size(i) > 1]
            if any(a < b for a, b in zip(strides[:-1], strides[1:])):
                warnings.warn("low-performance memory layout detected in filtered_lrelu input", RuntimeWarning)

            # #region agent log
            torch.cuda.synchronize()
            _timings['contiguous'] = time.time() - _t0_contig
            _t0_kernel = time.time()
            # #endregion

            # Call C++/Cuda plugin if datatype is supported.
            if x.dtype in [torch.float16, torch.float32, torch.bfloat16]:
                y, so, return_code = _plugin.filtered_lrelu(x, fu, fd, b, si, up, down, px0, px1, py0, py1, sx, sy, gain, slope, clamp, flip_filter, write_signs)
                # #region agent log
                torch.cuda.synchronize()
                _timings['cuda_kernel'] = time.time() - _t0_kernel
                # #endregion
            else:
                return_code = -1

            # No Cuda kernel found? Fall back to generic implementation. Still more memory efficient than the reference implementation because
            # only the bit-packed sign tensor is retained for gradient computation.
            if return_code < 0:
                warnings.warn("filtered_lrelu called with parameters that have no optimized CUDA kernel, using generic fallback", RuntimeWarning)
                
                # #region agent log
                _t0_fallback = time.time()
                # #endregion

                y = x.add(b.unsqueeze(-1).unsqueeze(-1)) # Add bias.
                
                # #region agent log
                torch.cuda.synchronize()
                _timings['fallback_bias'] = time.time() - _t0_fallback
                _t0_up = time.time()
                # #endregion
                
                y = upfirdn2d.upfirdn2d(x=y, f=fu, up=up, padding=[px0, px1, py0, py1], gain=up**2, flip_filter=flip_filter) # Upsample.
                
                # #region agent log
                torch.cuda.synchronize()
                _timings['fallback_upsample'] = time.time() - _t0_up
                _t0_act = time.time()
                # #endregion
                
                so = _plugin.filtered_lrelu_act_(y, si, sx, sy, gain, slope, clamp, write_signs) # Activation function and sign handling. Modifies y in-place.
                
                # #region agent log
                torch.cuda.synchronize()
                _timings['fallback_activation'] = time.time() - _t0_act
                _t0_down = time.time()
                # #endregion
                
                y = upfirdn2d.upfirdn2d(x=y, f=fd, down=down, flip_filter=flip_filter) # Downsample.
                
                # #region agent log
                torch.cuda.synchronize()
                _timings['fallback_downsample'] = time.time() - _t0_down
                # #endregion

            # Prepare for gradient computation.
            ctx.save_for_backward(fu, fd, (si if si.numel() else so))
            ctx.x_shape = x.shape
            ctx.y_shape = y.shape
            ctx.s_ofs = sx, sy
            
            # #region agent log
            torch.cuda.synchronize()
            kernel_time = time.time() - kernel_start
            _timings['total'] = kernel_time
            
            # Log EVERY call to identify where time is spent
            if _kernel_call_count <= 200:  # Log first 200 kernel calls
                _debug_log("filtered_lrelu.py:forward", "Kernel execution timing", {
                    "config_key": config_key,
                    "call_count": _kernel_call_count,
                    "x_shape": list(x.shape),
                    "y_shape": list(y.shape),
                    "up": up,
                    "down": down,
                    "return_code": return_code,
                    "timings": _timings,
                    "used_fallback": return_code < 0,
                    "tensor_bytes_gb": (x.numel() * x.element_size()) / (1024**3)
                }, "B")
            
            # Log only if kernel takes unusually long (>0.5 sec) - indicates JIT compilation
            if kernel_time > 0.5:
                _debug_log("filtered_lrelu.py:forward", "SLOW kernel execution detected!", {
                    "config_key": config_key,
                    "kernel_time_sec": kernel_time,
                    "x_shape": list(x.shape),
                    "y_shape": list(y.shape),
                    "return_code": return_code if 'return_code' in dir() else "N/A",
                    "call_count": _kernel_call_count,
                    "timings": _timings
                }, "C")
            
            # Track first call to each config
            if config_key not in _kernel_config_times:
                _kernel_config_times[config_key] = kernel_time
                _debug_log("filtered_lrelu.py:forward", "First call to kernel config", {
                    "config_key": config_key,
                    "kernel_time_sec": kernel_time,
                    "call_count": _kernel_call_count,
                    "timings": _timings
                }, "D")
            # #endregion
            
            return y

        @staticmethod
        def backward(ctx, dy): # pylint: disable=arguments-differ
            fu, fd, si = ctx.saved_tensors
            _, _, xh, xw = ctx.x_shape
            _, _, yh, yw = ctx.y_shape
            sx, sy = ctx.s_ofs
            dx  = None # 0
            dfu = None; assert not ctx.needs_input_grad[1]
            dfd = None; assert not ctx.needs_input_grad[2]
            db  = None # 3
            dsi = None; assert not ctx.needs_input_grad[4]
            dsx = None; assert not ctx.needs_input_grad[5]
            dsy = None; assert not ctx.needs_input_grad[6]

            if ctx.needs_input_grad[0] or ctx.needs_input_grad[3]:
                pp = [
                    (fu.shape[-1] - 1) + (fd.shape[-1] - 1) - px0,
                    xw * up - yw * down + px0 - (up - 1),
                    (fu.shape[0] - 1) + (fd.shape[0] - 1) - py0,
                    xh * up - yh * down + py0 - (up - 1),
                ]
                gg = gain * (up ** 2) / (down ** 2)
                ff = (not flip_filter)
                sx = sx - (fu.shape[-1] - 1) + px0
                sy = sy - (fu.shape[0]  - 1) + py0
                dx = _filtered_lrelu_cuda(up=down, down=up, padding=pp, gain=gg, slope=slope, clamp=None, flip_filter=ff).apply(dy, fd, fu, None, si, sx, sy)

            if ctx.needs_input_grad[3]:
                db = dx.sum([0, 2, 3])

            return dx, dfu, dfd, db, dsi, dsx, dsy

    # Add to cache.
    _filtered_lrelu_cuda_cache[key] = FilteredLReluCuda
    return FilteredLReluCuda

#----------------------------------------------------------------------------
