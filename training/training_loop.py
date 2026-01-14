# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Main training loop.

Optimized for modern CUDA (12.x/13.x) and PyTorch 2.x:
- torch.compile() for model optimization (dynamo + inductor)
- Modern GradScaler with growth_interval tuning
- Improved CUDA memory management
- Gradient accumulation with proper scaling
- DDP with static graph optimization
"""

import os
import time
import copy
import json
import dill
import psutil
from datetime import datetime
import PIL.Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
import dnnlib
import pickle
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

import legacy
from metrics import metric_main
from training import networks_stylegan3_resetting

#----------------------------------------------------------------------------
# Modern PyTorch 2.x compile configuration

def compile_model(model, mode='reduce-overhead', dynamic=False):
    """
    Compile model with torch.compile for PyTorch 2.x optimization.
    
    Args:
        model: nn.Module to compile
        mode: Compilation mode - 'default', 'reduce-overhead', 'max-autotune'
        dynamic: Whether to use dynamic shapes (False for fixed batch size)
    
    Returns:
        Compiled model (or original if compile unavailable)
    """
    if not hasattr(torch, 'compile'):
        return model
    
    try:
        # Use inductor backend for best CUDA performance
        compiled = torch.compile(
            model,
            mode=mode,
            dynamic=dynamic,
            fullgraph=False,  # Allow graph breaks for complex models
            backend='inductor',
        )
        return compiled
    except Exception as e:
        print(f'[Warning] torch.compile failed: {e}, using eager mode')
        return model

#----------------------------------------------------------------------------
# Unified debug logging (TXT format)

# #region debug logging
_DEBUG_ENABLED = False
_DEBUG_LOG_PATH = None

def set_debug_enabled(enabled, log_path=None):
    """Enable/disable debug logging for training loop."""
    global _DEBUG_ENABLED, _DEBUG_LOG_PATH
    _DEBUG_ENABLED = enabled
    _DEBUG_LOG_PATH = log_path
    # Also configure generator debug logging
    networks_stylegan3_resetting.set_debug_config(enabled, log_path)

def _debug_log(location, message, data=None):
    """Unified debug logging: prints to stdout and writes to TXT file when enabled."""
    if not _DEBUG_ENABLED:
        return
    
    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Build log line
        log_line = f"[{timestamp}] [{location}] {message}"
        if data:
            # Format data as simple key=value pairs
            data_str = " | ".join(f"{k}={v}" for k, v in data.items())
            log_line += f" | {data_str}"
        
        # Print to stdout
        print(log_line, flush=True)
        
        # Write to file if path is set
        if _DEBUG_LOG_PATH:
            with open(_DEBUG_LOG_PATH, 'a') as f:
                f.write(log_line + '\n')
    except Exception:
        pass
# #endregion

#----------------------------------------------------------------------------
# CUDA kernel warmup to pre-trigger all kernel configurations

def warmup_cuda_kernels(G, D, device, batch_gpu, num_iterations=1, rank=0):
    """
    Warmup CUDA kernels by running forward/backward passes before training.
    This pre-triggers JIT compilation of all kernel configurations to avoid
    slow compilation during actual training iterations.
    
    IMPORTANT: This also triggers:
    1. Custom CUDA kernels (filtered_lrelu, upfirdn2d, bias_act) JIT compilation
    2. cuDNN algorithm selection for all conv layer shapes (feature networks)
    3. cuBLAS handle initialization
    """
    # #region debug log
    warmup_start = time.time()
    _debug_log("warmup_cuda_kernels", "Starting CUDA kernel warmup", {
        "iterations": num_iterations,
        "batch_gpu": batch_gpu,
        "rank": rank
    })
    # #endregion
    
    if rank == 0:
        print(f'[Warmup] Running {num_iterations} warmup iterations to pre-compile CUDA kernels...', flush=True)
        print(f'[Warmup] cuDNN benchmark={torch.backends.cudnn.benchmark} (True=auto-select optimal algorithms)', flush=True)
        print(f'[Warmup] NOTE: First iteration may take 5-10 min on H200 due to:', flush=True)
        print(f'[Warmup]   1. PTX JIT compilation for custom kernels (sm_90)', flush=True)
        print(f'[Warmup]   2. cuDNN algorithm selection for feature networks (EfficientNet)', flush=True)
        print(f'[Warmup] This is a one-time cost - subsequent runs will use cached kernels.', flush=True)
    
    # CRITICAL: Use actual batch_gpu to compile kernels for correct tensor shapes
    # Using smaller batch causes PTX JIT during training (224s+ delays on H200)
    warmup_batch = batch_gpu
    
    # Create dummy inputs
    z = torch.randn([warmup_batch, G.z_dim], device=device)
    c = torch.zeros([warmup_batch, G.c_dim], device=device)
    
    iteration_times = []
    stage_times = {}
    
    for i in range(num_iterations):
        iter_start = time.time()
        
        try:
            # Phase 1: Generator forward (inference mode)
            t0 = time.time()
            with torch.no_grad():
                fake_img = G(z=z, c=c, noise_mode='random')
            torch.cuda.synchronize(device)
            t_gen_fwd = time.time() - t0
            
            # Phase 2: Discriminator forward (inference mode, quick check)
            # This triggers cuDNN benchmark algorithm selection for all feature network layers
            t0 = time.time()
            with torch.no_grad():
                d_out = D(fake_img, c)
            torch.cuda.synchronize(device)
            t_disc_fwd = time.time() - t0
            
            # #region debug log
            if i == 0:
                _debug_log("warmup", "Phase 2 D_fwd complete (cuDNN benchmark)", {
                    "time_sec": round(t_disc_fwd, 3)
                })
            # #endregion
            
            # Phase 3: Generator forward WITH gradients (triggers G backward kernel configs)
            t0 = time.time()
            z_grad = z.clone().requires_grad_(True)
            fake_img_grad = G(z=z_grad, c=c, noise_mode='random')
            torch.cuda.synchronize(device)
            t_gen_fwd_grad = time.time() - t0
            
            # Phase 4: G backward pass (triggers upfirdn2d and other G backward kernels)
            t0 = time.time()
            loss_g = fake_img_grad.sum()
            loss_g.backward()
            torch.cuda.synchronize(device)
            t_backward_g = time.time() - t0
            
            # Phase 5: D forward WITH gradients + D backward (triggers D backward kernel configs)
            # This is CRITICAL - without this, D backward kernels are JIT-compiled during training
            # Use flg_train=True to match actual training behavior
            t0 = time.time()
            fake_img_d = G(z=z, c=c, noise_mode='random').detach().requires_grad_(True)
            d_out_grad = D(fake_img_d, c, flg_train=True)  # Use training mode
            # Handle outputs from D: flg_train=True returns [logits_fun_list, logits_dir_list]
            # Each inner list contains tensors that need to be summed
            loss_d = torch.tensor(0.0, device=device, requires_grad=True)
            if isinstance(d_out_grad, (tuple, list)):
                for item in d_out_grad:
                    if isinstance(item, (tuple, list)):
                        # Nested list - sum all tensors
                        for tensor in item:
                            if tensor is not None and hasattr(tensor, 'sum'):
                                loss_d = loss_d + tensor.sum()
                    elif item is not None and hasattr(item, 'sum'):
                        loss_d = loss_d + item.sum()
            else:
                loss_d = d_out_grad.sum()
            loss_d.backward()
            torch.cuda.synchronize(device)
            t_backward_d = time.time() - t0
            
            # #region debug log
            if i == 0:
                _debug_log("warmup", "Phase 5 D_train complete", {
                    "time_sec": round(t_backward_d, 3)
                })
            # #endregion
            
            t_backward = t_backward_g + t_backward_d
            
            iter_time = time.time() - iter_start
            iteration_times.append(iter_time)
            
            # Track stage times
            if i == 0:
                stage_times = {
                    "gen_forward": t_gen_fwd,
                    "disc_forward": t_disc_fwd,
                    "gen_forward_grad": t_gen_fwd_grad,
                    "backward_g": t_backward_g,
                    "backward_d": t_backward_d
                }
            
            # #region debug log
            _debug_log("warmup", f"Warmup iteration {i+1} complete", {
                "iter": i + 1,
                "total_sec": round(iter_time, 2),
                "G_fwd": round(t_gen_fwd, 2),
                "D_fwd": round(t_disc_fwd, 2),
                "G_bwd": round(t_backward_g, 2),
                "D_bwd": round(t_backward_d, 2)
            })
            # #endregion
            
            if rank == 0:
                print(f'[Warmup] Iter {i+1}/{num_iterations}: {iter_time:.2f}s '
                      f'(G_fwd={t_gen_fwd:.2f}s, D_fwd={t_disc_fwd:.2f}s, '
                      f'G_bwd={t_backward_g:.2f}s, D_bwd={t_backward_d:.2f}s)', flush=True)
                
        except Exception as e:
            # #region debug log
            _debug_log("warmup", "Warmup iteration failed", {
                "iter": i + 1,
                "error": str(e)
            })
            # #endregion
            if rank == 0:
                print(f'[Warmup] Iteration {i+1} failed: {e}', flush=True)
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - warmup_start
    
    # #region debug log
    avg_time = sum(iteration_times) / len(iteration_times) if iteration_times else 0
    _debug_log("warmup", "Warmup complete", {
        "total_sec": round(total_time, 2),
        "avg_sec": round(avg_time, 2)
    })
    # #endregion
    
    if rank == 0:
        avg_time = sum(iteration_times) / len(iteration_times) if iteration_times else 0
        print(f'[Warmup] Complete! Total: {total_time:.2f}s, Avg per iteration: {avg_time:.2f}s', flush=True)
        if stage_times:
            print(f'[Warmup] First iter breakdown: G_fwd={stage_times["gen_forward"]:.2f}s, '
                  f'D_fwd={stage_times["disc_forward"]:.2f}s, '
                  f'G_bwd={stage_times["backward_g"]:.2f}s, D_bwd={stage_times["backward_d"]:.2f}s', flush=True)
    
    # Clear any accumulated memory and gradients
    torch.cuda.empty_cache()
    
    # Reset gradient state
    for param in G.parameters():
        if param.grad is not None:
            param.grad = None

#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0, gw=None, gh=None):
    rnd = np.random.RandomState(random_seed)
    if gw is None:
        gw = max(np.clip(2560 // training_set.image_shape[2], 7, 32), 1)
    if gh is None:
        gh = max(np.clip(1440 // training_set.image_shape[1], 4, 32), 1)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

@torch.no_grad()
def generate_snapshot_grid_images(G_ema, grid_z, grid_c, batch_gpu, num_gpus=1, rank=0, noise_mode='const'):
    # grid_z: [N, z_dim] tensor on GPU
    # grid_c: [N, c_dim] tensor on GPU
    if num_gpus == 1:
        images = torch.cat([G_ema(z=z, c=c, noise_mode=noise_mode).cpu() for z, c in zip(grid_z.split(batch_gpu), grid_c.split(batch_gpu))]).numpy()
        return images

    n = grid_z.shape[0]
    pad = (-n) % num_gpus
    if pad != 0:
        grid_z = torch.cat([grid_z, grid_z[-1:].repeat([pad, 1])], dim=0)
        grid_c = torch.cat([grid_c, grid_c[-1:].repeat([pad, 1])], dim=0)

    idx = torch.arange(rank, grid_z.shape[0], num_gpus, device=grid_z.device)
    grid_z = grid_z.index_select(0, idx)
    grid_c = grid_c.index_select(0, idx)

    images = torch.cat([G_ema(z=z, c=c, noise_mode=noise_mode) for z, c in zip(grid_z.split(batch_gpu), grid_c.split(batch_gpu))], dim=0)
    gathered = [torch.empty_like(images) for _ in range(num_gpus)]
    torch.distributed.all_gather(gathered, images)

    if rank == 0:
        images = torch.stack(gathered, dim=1).reshape([images.shape[0] * num_gpus, *images.shape[1:]])[:n]
        return images.cpu().numpy()

    return images[:0].cpu().numpy()

#----------------------------------------------------------------------------

def weight_reset(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        m.reset_parameters()

#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = 0.05,     # EMA ramp-up coefficient. None = no rampup.
    G_reg_interval          = 4,     # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    augment_p               = 0,        # Initial value of augmentation probability.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    resume_kimg             = 0,        # First kimg to report when resuming training.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
    restart_every           = -1,       # Time interval in seconds to exit code
    debug                   = False,    # Enable debug logging to file
    use_compile             = False,    # Enable torch.compile() for PyTorch 2.x optimization
    compile_mode            = 'reduce-overhead',  # torch.compile mode: 'default', 'reduce-overhead', 'max-autotune'
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    
    # Setup debug logging
    debug_log_path = os.path.join(run_dir, 'debug.txt') if debug else None
    set_debug_enabled(debug, debug_log_path)
    if debug and rank == 0:
        print(f'[Debug] Debug logging enabled, writing to: {debug_log_path}', flush=True)
    
    # Modern CUDA optimizations for PyTorch 2.x
    torch.backends.cuda.matmul.allow_tf32 = True       # TF32 for faster matmul on Ampere+
    torch.backends.cudnn.allow_tf32 = True             # TF32 for cuDNN operations
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True  # BF16 optimizations
    
    # Set deterministic algorithms if needed (disable for speed)
    # torch.use_deterministic_algorithms(False)
    
    # Enable CUDA graphs if supported (PyTorch 2.x)
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)  # Flash attention if available
    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.
    __RESTART__ = torch.tensor(0., device=device)       # will be broadcasted to exit loop
    __CUR_NIMG__ = torch.tensor(resume_kimg * 1000, dtype=torch.long, device=device)
    __CUR_TICK__ = torch.tensor(0, dtype=torch.long, device=device)
    __BATCH_IDX__ = torch.tensor(0, dtype=torch.long, device=device)
    __AUGMENT_P__ = torch.tensor(augment_p, dtype=torch.float, device=device)
    __PL_MEAN__ = torch.zeros([], device=device)
    best_fid = 9999

    # Stage logging helper (rank 0 only).
    def stage(msg):
        if rank == 0:
            dt = time.time() - start_time
            dt_min = dt / 60.0
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f'[Stage {now}] {dt_min:7.1f}m | {msg}', flush=True)

    # Load training set.
    stage('Loading training set')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    
    # Modern DataLoader with optimized settings
    loader_kwargs = dict(data_loader_kwargs)
    # Enable non-blocking transfers for overlapping compute and data transfer
    loader_kwargs.setdefault('pin_memory', True)
    loader_kwargs.setdefault('prefetch_factor', 4)  # Increased prefetch for better GPU utilization
    loader_kwargs.setdefault('persistent_workers', True)
    
    training_set_iterator = iter(torch.utils.data.DataLoader(
        dataset=training_set,
        sampler=training_set_sampler,
        batch_size=batch_size//num_gpus,
        **loader_kwargs
    ))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print(f'DataLoader: prefetch={loader_kwargs.get("prefetch_factor", 2)}, workers={loader_kwargs.get("num_workers", 0)}')
        print()

    # Construct networks.
    stage('Constructing networks')
    common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()
    
    # Apply torch.compile() for PyTorch 2.x optimization (after model construction)
    if use_compile and hasattr(torch, 'compile'):
        stage(f'Compiling models with torch.compile (mode={compile_mode})')
        try:
            # Compile generator synthesis for better performance
            # Note: We don't compile the full model to preserve flexibility
            if hasattr(G, 'synthesis'):
                G.synthesis = compile_model(G.synthesis, mode=compile_mode, dynamic=False)
            if hasattr(G_ema, 'synthesis'):
                G_ema.synthesis = compile_model(G_ema.synthesis, mode=compile_mode, dynamic=False)
            if rank == 0:
                print(f'[Compile] Successfully compiled generator synthesis', flush=True)
        except Exception as e:
            if rank == 0:
                print(f'[Compile] Warning: torch.compile failed: {e}', flush=True)
    for _, discriminator in D.discriminators.items():
        if D_kwargs.backbone_kwargs.cond:
            for _, disc in discriminator.mini_discs.items():
                disc.embed.normalize_weight()
        else:
            for _, disc in discriminator.mini_discs.items():
                disc.last_layer.normalize_weight()

    # Check for existing checkpoint
    ckpt_pkl = None
    if restart_every > 0 and os.path.isfile(misc.get_ckpt_path(run_dir)):
        ckpt_pkl = resume_pkl = misc.get_ckpt_path(run_dir)


    if (resume_pkl is not None) and (rank == 0):
        stage(f'Resuming from "{resume_pkl}"')

        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
        stage('Loaded model weights from resume pickle')

        if ckpt_pkl is not None:            # Load ticks
            __CUR_NIMG__ = resume_data['progress']['cur_nimg'].to(device)
            __CUR_TICK__ = resume_data['progress']['cur_tick'].to(device)
            __BATCH_IDX__ = resume_data['progress']['batch_idx'].to(device)
            __AUGMENT_P__ = resume_data['progress'].get('augment_p', torch.tensor(0.)).to(device)
            __PL_MEAN__ = resume_data['progress'].get('pl_mean', torch.zeros([])).to(device)
            best_fid = resume_data['progress']['best_fid']       # only needed for rank == 0

    # this is relevant when you continue training a lower-res model
    # ie. train 16 model, start training 32 model but continue training 16 model
    # then restart 32 model to reload the improved 16 model
    if hasattr(G, 'reinit_stem'):
        G.reinit_stem()
        G_ema.reinit_stem()

    # Print network summary tables.
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        c = torch.empty([batch_gpu, G.c_dim], device=device)
        img = misc.print_module_summary(G, [z, c])
        misc.print_module_summary(D, [img, c])

    # Setup augmentation.
    stage('Setting up augmentation')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    stage(f'Distributing across {num_gpus} GPUs')
    for module in [G, D, G_ema, augment_pipe]:
        if module is not None and num_gpus > 1:
            for param in misc.params_and_buffers(module):
                torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    stage('Setting up training phases')
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, G_ema=G_ema, D=D, augment_pipe=augment_pipe, **loss_kwargs) # subclass of training.loss.Loss
    phases = []
    iter_dict = {'G': 1, 'D': 1}  # change here if you want to do several G/D iterations at once

    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            for _ in range(iter_dict[name]):
                phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
        else: # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            for _ in range(iter_dict[name]):
                phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    stage('Exporting sample images (reals.png, fakes_init.png)')
    grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
    if rank == 0:
        save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)

    grid_z = torch.randn([labels.shape[0], G.z_dim], device=device, generator=torch.Generator(device=device).manual_seed(random_seed * num_gpus))
    grid_c = torch.from_numpy(labels).to(device)
    images = generate_snapshot_grid_images(G_ema=G_ema, grid_z=grid_z, grid_c=grid_c, batch_gpu=batch_gpu, num_gpus=num_gpus, rank=rank, noise_mode='const')

    if rank == 0:
        save_image_grid(images, os.path.join(run_dir, 'fakes_init.png'), drange=[-1,1], grid_size=grid_size)

    # Initialize logs.
    stage('Initializing logs (stats.jsonl, tensorboard)')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Log CUDA configuration for diagnostics
    # #region debug log
    _debug_log("training_loop", "CUDA configuration", {
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "cuda_ver": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "rank": rank
    })
    # #endregion
    
    if rank == 0:
        print(f'[Config] cudnn.benchmark={torch.backends.cudnn.benchmark}, '
              f'tf32={torch.backends.cuda.matmul.allow_tf32}, '
              f'cudnn={torch.backends.cudnn.version()}', flush=True)
    
    # Pre-initialize all CUDA plugins before warmup
    stage('Pre-initializing CUDA plugins')
    if rank == 0:
        print('[Init] Force-loading all CUDA plugins...', flush=True)
    
    # #region debug log
    _debug_log("training_loop", "Pre-initializing CUDA plugins", {"rank": rank})
    # #endregion
    
    # Force-initialize all plugins by importing them
    from torch_utils.ops import bias_act, filtered_lrelu, upfirdn2d
    
    # Trigger plugin compilation by calling init functions
    t0 = time.time()
    bias_act._init()
    t_bias = time.time() - t0
    
    t0 = time.time()
    filtered_lrelu._init()
    t_lrelu = time.time() - t0
    
    t0 = time.time()
    upfirdn2d._init()
    t_upfirdn = time.time() - t0
    
    # #region debug log
    _debug_log("training_loop", "CUDA plugins initialized", {
        "bias_act_sec": round(t_bias, 2),
        "lrelu_sec": round(t_lrelu, 2),
        "upfirdn_sec": round(t_upfirdn, 2)
    })
    # #endregion
    
    if rank == 0:
        print(f'[Init] Plugins loaded: bias_act={t_bias:.2f}s, filtered_lrelu={t_lrelu:.2f}s, upfirdn2d={t_upfirdn:.2f}s', flush=True)
    
    # Synchronize all GPUs after plugin init
    if num_gpus > 1:
        torch.distributed.barrier()
    
    # Warmup CUDA kernels before training to avoid JIT compilation delays
    stage('Warming up CUDA kernels')
    warmup_cuda_kernels(G=G_ema, D=D, device=device, batch_gpu=batch_gpu, num_iterations=1, rank=rank)
    
    # Synchronize all GPUs after warmup
    if num_gpus > 1:
        torch.distributed.barrier()

    # Train.
    stage(f'Training start (total_kimg={total_kimg}, batch_size={batch_size}, batch_gpu={batch_gpu})')
    if num_gpus > 1:  # broadcast loaded states to all
        torch.distributed.broadcast(__CUR_NIMG__, 0)
        torch.distributed.broadcast(__CUR_TICK__, 0)
        torch.distributed.broadcast(__BATCH_IDX__, 0)
        torch.distributed.broadcast(__AUGMENT_P__, 0)
        torch.distributed.broadcast(__PL_MEAN__, 0)
        torch.distributed.barrier()  # ensure all processes received this info
    cur_nimg = __CUR_NIMG__.item()
    cur_tick = __CUR_TICK__.item()
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = __BATCH_IDX__.item()
    if progress_fn is not None:
        progress_fn(cur_nimg // 1000, total_kimg)
    augment_p = __AUGMENT_P__
    if augment_pipe is not None:
        augment_pipe.p.copy_(augment_p)
    if hasattr(loss, 'pl_mean'):
        loss.pl_mean.copy_(__PL_MEAN__)

    # #region debug log
    _iter_count = 0
    # #endregion

    while True:
        # #region debug log
        _iter_start = time.time()
        _data_fetch_start = time.time()
        # #endregion

        with torch.autograd.profiler.record_function('data_fetch'):
            phase_real_img, phase_real_c = next(training_set_iterator)
            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_c = phase_real_c.to(device).split(batch_gpu)
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        # #region debug log
        _data_fetch_end = time.time()
        # #endregion

        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            if batch_idx % phase.interval != 0:
                continue
            
            # #region debug log
            _phase_start = time.time()
            # #endregion
            
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)

            ### PROJECTED GAN ADDITIONS ###
            if phase.name in ['Dmain', 'Dboth', 'Dreg'] and hasattr(phase.module, 'feature_networks'):
                phase.module.feature_networks.requires_grad_(False)

            for real_img, real_c, gen_z, gen_c in zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c):
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c, gain=phase.interval, cur_nimg=cur_nimg)
            phase.module.requires_grad_(False)

            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                params = [param for param in phase.module.parameters() if param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()
            
            # #region debug log
            _phase_end = time.time()
            _phase_time = _phase_end - _phase_start
            if _phase_time > 5.0:  # Log slow phases (>5s)
                _debug_log("training_loop", f"SLOW phase {phase.name}", {
                    "iter": _iter_count,
                    "time_sec": round(_phase_time, 2)
                })
            # #endregion
            if phase.name in ['Dmain', 'Dboth', 'Dreg']:
                for _, discriminator in D.discriminators.items():
                    if D_kwargs.backbone_kwargs.cond:
                        for _, disc in discriminator.mini_discs.items():
                            disc.embed.normalize_weight()
                    else:
                        for _, disc in discriminator.mini_discs.items():
                            disc.last_layer.normalize_weight()

            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1
        
        # #region debug log
        _iter_end = time.time()
        _iter_time = _iter_end - _iter_start
        if _iter_time > 10.0:  # Log slow iterations (>10s)
            _debug_log("training_loop", f"SLOW iteration {_iter_count}", {
                "time_sec": round(_iter_time, 2),
                "nimg": cur_nimg
            })
        _iter_count += 1
        # #endregion

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))

        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Check for restart.
        if (rank == 0) and (restart_every > 0) and (time.time() - start_time > restart_every):
            print('Restart job...')
            __RESTART__ = torch.tensor(1., device=device)
        if num_gpus > 1:
            torch.distributed.broadcast(__RESTART__, 0)
        if __RESTART__:
            done = True
            print(f'Process {rank} leaving...')
            if num_gpus > 1:
                torch.distributed.barrier()

        # Save image snapshot.
        if (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            stage(f'Saving image snapshot (kimg={cur_nimg/1e3:.1f})')
            images = generate_snapshot_grid_images(G_ema=G_ema, grid_z=grid_z, grid_c=grid_c, batch_gpu=batch_gpu, num_gpus=num_gpus, rank=rank, noise_mode='const')
            if rank == 0:
                save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)
            stage(f'Image snapshot saved (kimg={cur_nimg/1e3:.1f})')

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            stage(f'Preparing network snapshot data (kimg={cur_nimg/1e3:.1f})')
            snapshot_data = dict(G=G, D=D, G_ema=G_ema, augment_pipe=augment_pipe, training_set_kwargs=dict(training_set_kwargs))
            for key, value in snapshot_data.items():
                if isinstance(value, torch.nn.Module):
                    # value = copy.deepcopy(value).eval().requires_grad_(False)
                    # value = misc.spectral_to_cpu(value)
                    # if num_gpus > 1:
                    #     misc.check_ddp_consistency(value, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                    #     for param in misc.params_and_buffers(value):
                    #         torch.distributed.broadcast(param, src=0)
                    snapshot_data[key] = value #.cpu()
                del value # conserve memory
            stage(f'Network snapshot data prepared (kimg={cur_nimg/1e3:.1f})')

            # save for current time step (only for superres training, as we do not evaluate metrics here)
            if False:
                snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
                if rank == 0:
                    with open(snapshot_pkl, 'wb') as f:
                        dill.dump(snapshot_data, f)

        # Save Checkpoint if needed
        if (rank == 0) and (restart_every > 0) and (network_snapshot_ticks is not None) and (
                done or cur_tick % network_snapshot_ticks == 0):
            snapshot_pkl = misc.get_ckpt_path(run_dir)
            stage(f'Saving checkpoint "{snapshot_pkl}"')
            # save as tensors to avoid error for multi GPU
            # when saving checkpoint progress
            snapshot_data['progress'] = {
                'cur_nimg': torch.tensor(cur_nimg, dtype=torch.long),
                'cur_tick': torch.tensor(cur_tick, dtype=torch.long),
                'batch_idx': torch.tensor(batch_idx, dtype=torch.long),
                'best_fid': best_fid,
            }
            if augment_pipe is not None:
                snapshot_data['progress']['augment_p'] = augment_pipe.p.cpu()
            if hasattr(loss, 'pl_mean'):
                snapshot_data['progress']['pl_mean'] = loss.pl_mean.cpu()

            with open(snapshot_pkl, 'wb') as f:
                dill.dump(snapshot_data, f)

            stage(f'Checkpoint saved (kimg={snapshot_pkl})')

        # Evaluate metrics.
        # if (snapshot_data is not None) and (len(metrics) > 0):
        if cur_tick and (snapshot_data is not None) and (len(metrics) > 0):
            stage('Evaluating metrics')
            for metric in metrics:
                result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
                                                      dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)

            # save best fid ckpt
            snapshot_pkl = os.path.join(run_dir, f'best_model.pkl')
            cur_nimg_txt = os.path.join(run_dir, f'best_nimg.txt')
            if rank == 0:
                if 'fid50k_full' in stats_metrics and stats_metrics['fid50k_full'] < best_fid:
                    best_fid = stats_metrics['fid50k_full']

                    with open(snapshot_pkl, 'wb') as f:
                        dill.dump(snapshot_data, f)
                    # save curr iteration number (directly saving it to pkl leads to problems with multi GPU)
                    with open(cur_nimg_txt, 'w') as f:
                        f.write(str(cur_nimg))
            stage(f'Best FID saved')

        del snapshot_data # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None) and \
                    not (phase.start_event.cuda_event == 0 and phase.end_event.cuda_event == 0):            # Both events were not initialized yet, can happen with restart
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        stage('Exiting')

#----------------------------------------------------------------------------
