# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Main training loop."""

import copy
import importlib.util
import json
import os
import time
import warnings
from datetime import datetime

import numpy as np
import PIL.Image
import psutil
import torch

import dnnlib
from torch_utils import checkpoint, misc, training_stats
from torch_utils.ops import conv2d_gradfix, grid_sample_gradfix
from training import networks_stylegan3_resetting

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
    # Also configure loss debug logging (lazy import to avoid any import cycle)
    from training import loss as _loss_module
    _loss_module.set_debug_config(enabled, log_path)

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
        print('[Warmup] NOTE: First iteration may take 5-10 min on H200 due to:', flush=True)
        print('[Warmup]   1. PTX JIT compilation for custom kernels (sm_90)', flush=True)
        print('[Warmup]   2. cuDNN algorithm selection for feature networks (EfficientNet)', flush=True)
        print('[Warmup] This is a one-time cost - subsequent runs will use cached kernels.', flush=True)
    
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
                D(fake_img, c)
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

# Normalization contract (§5): san-v2's float training space is [-1, 1]. Exactly
# one denorm formula converts back to uint8 [0, 255], used for every artifact and
# every combra batch, so reals and fakes cross the boundary identically (the legacy
# eval path used a `+128` offset for fakes while reals entered as raw uint8, biasing
# every reported metric). ``x`` is a float array/tensor in [-1, 1]; NCHW is preserved.
def denorm_to_uint8(x):
    x = np.asarray(x.cpu() if isinstance(x, torch.Tensor) else x, dtype=np.float32)
    return np.rint((x + 1) * 127.5).clip(0, 255).astype(np.uint8)

# Inverse used by the training loader: uint8 [0, 255] -> float [-1, 1].
def norm_from_uint8(x):
    return x.to(torch.float32) / 127.5 - 1

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

def image_grid_array(img, drange, grid_size):
    """Tile a NCHW batch into one HWC uint8 grid. Shared by the PNG save and the
    TensorBoard `Fakes` image, so both show exactly the same pixels."""
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    return img.reshape([gh * H, gw * W, C])


def save_image_grid(img, fname, drange, grid_size):
    img = image_grid_array(img, drange, grid_size)
    C = img.shape[2]
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
# Distributed combra metrics: shard ALL the per-image extraction across ranks
# instead of running it on rank 0. Each rank generates its own shard of the eval
# fakes and, from that shard, extracts the image features (fid / cmmd / fd_dinov2)
# AND pools the vertex angles; the feature rows and the pooled-angle arrays are
# gathered to rank 0, which takes the Frechet / MMD distances and the angle
# Wasserstein / Gaussian metrics there against the (precomputed, cached) reference.
# The reference side is sharded the same way and extracted once before training, so
# no reference work or collective recurs per tick. Numerically identical to the
# single-GPU compute_all_metrics(image_metrics=True).
#
# This matters for correctness, not just speed: when the angle-density extraction
# ran rank-0-only over the full gathered set, at 512/1024 px it could take longer
# than NCCL's collective-timeout watchdog while the other ranks idled at the next
# allreduce, aborting the job.

@torch.no_grad()
def _combra_generate_local_shard(G_ema, grid_z, grid_c, batch_gpu, num_gpus, rank, noise_mode='const'):
    # Rank r generates the fakes at indices [r, r+num_gpus, ...]; concatenating every
    # rank's shard reproduces the full set exactly (no padding duplicates). G_ema is
    # deterministic given (z, c) with noise_mode='const', so which rank generates a
    # given latent is irrelevant. Returns a float [-1, 1] NCHW numpy array.
    n = grid_z.shape[0]
    idx = torch.arange(rank, n, num_gpus, device=grid_z.device)
    z = grid_z.index_select(0, idx)
    c = grid_c.index_select(0, idx)
    images = torch.cat([G_ema(z=zz, c=cc, noise_mode=noise_mode)
                        for zz, cc in zip(z.split(batch_gpu), c.split(batch_gpu))], dim=0)
    return images.cpu().numpy()


def _combra_precompute_reference(training_set, device, rank, num_gpus, ref_count=None, seed=0):
    """This rank's slice of the real reference, extracted and gathered by combra.

    ref_count caps the reference to a SEEDED RANDOM subset (§6) -- never the first N:
    dataset zips are class-sorted, so a first-N slice is class-biased. The same seed on
    every rank selects the same subset, then it is sharded by rank stride.

    Returns ``(reference, ok)``; ``ok`` is rank-uniform, so the caller can gate the
    per-tick eval on it (``reference`` is None on every non-zero rank regardless).
    """
    from combra.metrics.distributed import precompute_reference

    n = len(training_set)
    if ref_count is not None and ref_count < n:
        sel = np.sort(np.random.RandomState(seed).choice(n, size=ref_count, replace=False))
    else:
        sel = np.arange(n)
    idx = sel[rank::num_gpus]
    local_u8 = np.stack([training_set[i][0] for i in idx])  # NCHW uint8
    return precompute_reference(local_u8, device, rank, num_gpus)


def _combra_eval_distributed(G_ema, grid_z, grid_c, batch_gpu, num_gpus, rank, device,
                             combra_ref):
    """combra metrics on rank 0, None elsewhere. Every rank MUST call this: the
    gathers inside are collectives, so a rank that skips them hangs the others."""
    from combra.metrics.distributed import distributed_metrics, gather_generated

    # combra's angle path is scale-sensitive, so both sides cross the boundary as uint8
    # through the one denorm formula (§5).
    local_u8 = denorm_to_uint8(
        _combra_generate_local_shard(G_ema, grid_z, grid_c, batch_gpu, num_gpus, rank))
    gen_feats, gen_angles = gather_generated(local_u8, device, rank, num_gpus)
    if rank != 0:
        return None
    # device= so the CMMD reduction runs where the features were extracted.
    return distributed_metrics(combra_ref, gen_angles, gen_feats, device=device)


def _write_run_hparams(run_dir, writer, metrics):
    """Record this run's configuration in TensorBoard's HPARAMS tab (§7).

    The config is read back from ``training_options.json``, which the launcher has
    already written, so no plumbing through the training-loop signature is needed.
    Paired with the run's final metrics, this is what makes two runs comparable in
    TensorBoard: without it the curves are there but nothing says which
    configuration produced them, and comparing runs means diffing two JSON files.
    """
    import json
    import os

    try:
        from combra.io import write_hparams
    except ImportError:
        return  # combra is optional; the curves are still logged
    path = os.path.join(run_dir, 'training_options.json')
    if not os.path.isfile(path):
        return
    with open(path) as fh:
        config = json.load(fh)
    write_hparams(writer, config, metrics)


def build_stats_row(stats_dict, stats_metrics, timestamp, start_time):
    """One scalar JSON row for ``stats.jsonl`` (§7).

    Collector values are flattened to their means. The collector nests every
    scalar as ``{num, mean, std}``, and ``combra.metrics.load_fid_by_kimg`` reads
    ``Progress/kimg`` through ``int()`` -- a nested value makes it shape-filter the
    whole record away, so every run returned ``{}`` with no error and post-hoc
    snapshot selection silently found nothing. TensorBoard already receives
    ``value.mean``, so flattening also makes the file carry the same keys as the
    tags, which is what §7 asks for.

    Kept as a plain function so tests can exercise the real row without running a
    training loop.
    """
    row = {name: value.mean for name, value in stats_dict.items()}
    for name, value in stats_metrics.items():
        row[f'Metrics/{name}'] = float(value)
    row['timestamp'] = timestamp
    row['wall_time'] = timestamp - start_time
    row['datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return row


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
    snapshot_keep_last      = 3,        # How many inference snapshots to keep (0 = keep all).
    mirror                  = False,    # Stochastic per-item horizontal flip in the training loader (§5).
    combra_metrics          = True,     # Compute combra generative-quality metrics each snapshot tick.
    combra_num_gen          = 10000,    # Number of fakes for the combra metrics (0 disables eval).
    combra_ref_count        = None,     # Cap the combra reference to a seeded random subset.
    precision               = 'fp16',   # Training precision: fp32 / fp16 / bf16.
    allow_tf32              = True,      # Enable TF32 matmul / cuDNN.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
    debug                   = False,    # Enable debug logging to file (unused; retained for the warmup helpers).
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
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

    # Normalization contract (§5): assert the one normalize/denormalize pair round-trips
    # exactly, so every artifact and every combra batch crosses the uint8 boundary
    # losslessly.
    if rank == 0:
        _probe = torch.randint(0, 256, [4, 3, 8, 8], dtype=torch.uint8)
        assert np.array_equal(denorm_to_uint8(norm_from_uint8(_probe)), _probe.numpy()), \
            'normalize/denormalize pair must round-trip uint8 exactly'

    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.
    __CUR_NIMG__ = torch.tensor(0, dtype=torch.long, device=device)
    __CUR_TICK__ = torch.tensor(0, dtype=torch.long, device=device)
    __BATCH_IDX__ = torch.tensor(0, dtype=torch.long, device=device)
    __AUGMENT_P__ = torch.tensor(augment_p, dtype=torch.float, device=device)
    __PL_MEAN__ = torch.zeros([], device=device)

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
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()

    # Construct networks.
    stage('Constructing networks')
    common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()
    for _, discriminator in D.discriminators.items():
        if D_kwargs.backbone_kwargs.cond:
            for _, disc in discriminator.mini_discs.items():
                disc.embed.normalize_weight()
        else:
            for _, disc in discriminator.mini_discs.items():
                disc.last_layer.normalize_weight()

    # No resume (§3): training runs start-to-finish; a fresh run id is always
    # allocated and no checkpoint is loaded here. The only warm start is the
    # progressive stem, loaded weights-only inside SuperresGenerator via --path-stem.

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

    # Grid/eval latents derive from --seed ALONE (§2), never scaled by the GPU count,
    # so the same seed draws the same latents at any --gpus.
    grid_z = torch.randn([labels.shape[0], G.z_dim], device=device, generator=torch.Generator(device=device).manual_seed(random_seed))
    grid_c = torch.from_numpy(labels).to(device)
    images = generate_snapshot_grid_images(G_ema=G_ema, grid_z=grid_z, grid_c=grid_c, batch_gpu=batch_gpu, num_gpus=num_gpus, rank=rank, noise_mode='const')

    if rank == 0:
        save_image_grid(images, os.path.join(run_dir, 'fakes_init.png'), drange=[-1,1], grid_size=grid_size)

    # combra metric reference: the training set, scored against combra_num_gen
    # generated images (fid10k / cmmd10k / fd_dinov2_10k -> N fakes vs the reals;
    # the image metrics estimate per-side statistics, so the counts need not match).
    # --num-fid-samples=0 disables the combra eval entirely. --combra-ref-count caps
    # the reference to a seeded random subset (§6). Built once (the reference is fixed
    # across training): every rank extracts the angles + features for its deterministic
    # slice of the reals and the rows are gathered to rank 0, so the expensive reference
    # extraction is sharded (not rank-0-only) and the cached result is reused every
    # tick. Latents/labels live on every rank for distributed generation.
    combra_enabled = combra_metrics and combra_num_gen > 0
    combra_num = combra_num_gen
    # Labels for the fakes: sample the training-set label distribution with
    # replacement (seeded identically on every rank so all ranks generate the same
    # batch). For an unconditional G, get_label returns a zero-length vector.
    combra_ref = None
    combra_ref_ok = True
    if combra_enabled and (rank == 0) and (importlib.util.find_spec('combra') is not None):
        # Startup smoke test (§6): fail fast if the combra backends are unusable, using
        # combra's shared implementation. strict=True is the point -- the plain
        # self_test() this used to call defaults to image_metrics=False, so it never
        # touched InceptionV3 / CLIP / DINOv2 and a missing CLIP download surfaced only
        # as a whole run logging nan for combra_fid / combra_cmmd / combra_fd_dinov2.
        # images= scores the real reference slice, not synthetic grains.
        try:
            from combra.metrics import self_test
            sample = np.stack([training_set[i][0] for i in range(min(4, len(training_set)))])
            self_test(images=sample, device=device, image_metrics=True, strict=True)
        except ImportError:
            pass
        except Exception as e:
            print(f'[combra] smoke test reported a problem: {e}', flush=True)
    if combra_enabled:
        combra_label_idx = np.random.RandomState(random_seed).randint(0, len(training_set), size=combra_num)
        combra_labels = np.stack([training_set.get_label(i) for i in combra_label_idx])
        combra_c = torch.from_numpy(combra_labels).to(device)
        # Eval latents derive from --seed alone (§2), never scaled by the GPU count.
        combra_z = torch.randn([combra_num, G.z_dim], device=device,
            generator=torch.Generator(device=device).manual_seed(random_seed + 1))
    # combra_active must be rank-UNIFORM (it gates the precompute collectives, just
    # like the per-tick eval): combra_metrics is broadcast config and find_spec is a
    # rank-independent filesystem check. Runs on every rank so the gathers stay matched.
    if combra_enabled and (importlib.util.find_spec('combra') is not None):
        # A broken combra backend (e.g. no network for DINOv2/CLIP weights) fails the
        # same way on every rank -- before any gather in the loop below -- so the except
        # is reached symmetrically and leaves combra_ref None on all ranks rather than
        # half-completing a collective. Per-tick eval then no-ops via its own guard, so
        # an unusable combra never breaks training.
        try:
            combra_ref, combra_ref_ok = _combra_precompute_reference(
                training_set, device, rank, num_gpus, ref_count=combra_ref_count, seed=random_seed)
        except Exception as e:
            combra_ref_ok = False
            if rank == 0:
                print(f'[combra] reference precompute failed, disabling combra metrics: {e}', flush=True)

    # Warn early (once, on rank 0) if combra metrics are requested but the package is
    # missing, so the user is not left wondering why no combra_* values ever appear.
    if combra_enabled and (rank == 0) and (importlib.util.find_spec('combra') is None):
        warnings.warn(
            'combra_metrics=True but the `combra` package is not installed -- combra '
            'metrics will be skipped. Install it (e.g. `pip install -e ../wc_cv/combra`) '
            'to enable them, or pass --combra-metrics=false to silence this warning.')

    # Initialize logs.
    stage('Initializing logs (stats.jsonl, tensorboard)')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    best_fid = float('inf')   # running best combra FID -> Metrics/combra_fid_best
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            # filename_suffix carries the run name (§7) so a tfevents file copied out of
            # its run directory stays self-identifying -- which is exactly how the
            # wc_cv/ml/ event files are stored.
            stats_tfevents = tensorboard.SummaryWriter(
                run_dir, filename_suffix=f'.{os.path.basename(run_dir)}')
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
            phase_real_img = phase_real_img.to(device).to(torch.float32)
            # --mirror (§5): stochastic per-item horizontal flip, in the TRAINING loader
            # only. The eval / combra-reference / grid loaders read the dataset directly
            # and never flip, and the dataset is never flip-doubled.
            if mirror:
                flip = torch.rand([phase_real_img.shape[0], 1, 1, 1], device=device) < 0.5
                phase_real_img = torch.where(flip, phase_real_img.flip(-1), phase_real_img)
            phase_real_img = (phase_real_img / 127.5 - 1).split(batch_gpu)
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
        # §7 puts learning rates under LearningRate/. Reported per network rather than
        # per phase, since Dmain/Dreg share one optimizer.
        for _phase in phases:
            _tag = 'G' if _phase.name.startswith('G') else 'D'
            training_stats.report0(f'LearningRate/{_tag}', _phase.opt.param_groups[0]['lr'])
        # total_hours / total_days were san-v2-only and are just Timing/total_sec
        # rescaled; dropped so the four repos publish one set of Timing keys.

        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot (fakes grid). The last tick always snapshots (§3).
        if (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            stage(f'Saving image snapshot (kimg={cur_nimg/1e3:.1f})')
            images = generate_snapshot_grid_images(G_ema=G_ema, grid_z=grid_z, grid_c=grid_c, batch_gpu=batch_gpu, num_gpus=num_gpus, rank=rank, noise_mode='const')
            if rank == 0:
                save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)
                # §7 lists `Fakes` as a TensorBoard image every snapshot tick. This repo
                # is the reference implementation for the sample grid, yet it was the one
                # repo that only ever wrote the grid to disk.
                if stats_tfevents is not None:
                    stats_tfevents.add_image(
                        'Fakes', image_grid_array(images, [-1, 1], grid_size),
                        global_step=cur_nimg, dataformats='HWC')
            stage(f'Image snapshot saved (kimg={cur_nimg/1e3:.1f})')

        # Save network snapshot (§3): EMA-only weights as a `.pt` state dict, written
        # atomically every snapshot tick AND always at the last tick, so the newest
        # snapshot IS the final model. No resume/best/full checkpoints exist. History
        # is pruned to --snapshot-keep-last.
        did_snapshot = (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0)
        if did_snapshot:
            stage(f'Saving inference snapshot (kimg={cur_nimg/1e3:.1f})')
            # DDP weight-consistency check before saving so silently diverged ranks
            # (which would otherwise persist rank 0's weights undetected) are caught.
            # Must run on every rank.
            if num_gpus > 1:
                misc.check_ddp_consistency(G_ema, ignore_regex=r'.*\.[^.]+_(avg|ema)')
            if rank == 0:
                metadata = dict(
                    n_classes=int(training_set.label_dim),
                    resolution=int(training_set.resolution),
                    class_names=training_set.class_names,
                    cur_nimg=int(cur_nimg),
                )
                snap_path = os.path.join(run_dir, f'san-snapshot-{cur_nimg//1000:06d}-inference.pt')
                checkpoint.save_inference_snapshot(snap_path, G_ema, metadata)
                checkpoint.prune_snapshots(run_dir, snapshot_keep_last)
            stage(f'Inference snapshot saved (kimg={cur_nimg/1e3:.1f})')

        # combra in-memory generative-quality metrics (optional dependency), scored
        # over the reference vs combra_num_gen fakes. Both the image-feature extraction
        # (fid/cmmd/fd_dinov2) and the angle-density metrics are sharded across ranks
        # (each rank handles its own fakes); only the final distances run on rank 0.
        # A missing/uninstallable combra must never break training. Values are mirrored
        # into stats.jsonl (§6), not TensorBoard-only, so post-hoc snapshot selection
        # survives loss of the tfevents file.
        #
        # combra_active must be rank-UNIFORM: it gates the collectives inside
        # _combra_eval_distributed, so every rank must agree. It comes from the broadcast
        # config + a rank-independent filesystem check -- do NOT gate on combra_ref
        # (rank 0 only) or the gathers deadlock.
        # Clear the combra row first: the metrics are only recomputed at snapshot
        # ticks, so a dict that persists across ticks re-emits the previous tick's
        # values at a new step -- turning the metric curves into step functions and
        # letting post-hoc snapshot selection resolve to a kimg never evaluated.
        stats_metrics = {}
        # combra_ref_ok is rank-uniform (all_ranks_ok inside the precompute), so it is
        # safe to gate on: previously a failed precompute left combra_active True and
        # every tick then called the eval with combra_ref=None, raising inside the try.
        combra_active = (combra_enabled and combra_ref_ok
                         and (importlib.util.find_spec('combra') is not None))
        if cur_tick and did_snapshot and combra_active:
            stage('Evaluating combra metrics')
            eval_start = time.time()
            try:
                combra_results = _combra_eval_distributed(
                    G_ema, combra_z, combra_c, batch_gpu, num_gpus, rank, device, combra_ref)
            except Exception as e:
                combra_results = None
                if rank == 0:
                    print(f'[combra] metric evaluation failed: {e}', flush=True)
            if rank == 0 and combra_results is not None:
                # Cast to float so the scalar writers (Metrics/combra_*) accept every
                # value, including numpy scalars and NaNs. Keys are bare (combra_fid,
                # not combra_fid10k): the old `10k` suffix was a literal that stayed
                # 10k whatever --num-fid-samples said, so every chart built from it was
                # mislabelled. The count is logged once, as its own scalar.
                for name, value in combra_results.items():
                    stats_metrics[f'combra_{name}'] = float(value)
                stats_metrics['combra_num_fid_samples'] = float(combra_num)
                training_stats.report0('Timing/eval_sec', time.time() - eval_start)
                if 'combra_fid' in stats_metrics:
                    best_fid = min(best_fid, stats_metrics['combra_fid'])
                    stats_metrics['combra_fid_best'] = best_fid
                print('combra metrics: ' + ', '.join(
                    f'{k}={v:.4f}' for k, v in combra_results.items()), flush=True)

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

        # Update logs. combra metrics are mirrored into stats.jsonl under the same
        # Metrics/combra_* keys as TensorBoard (§6) -- stats.jsonl is the machine-
        # readable source of truth, so losing the tfevents file no longer loses the
        # run's FID/CMMD/angle history (and it is what enables post-hoc snapshot
        # selection). wall_time / datetime columns are added per §7.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = build_stats_row(stats_dict, stats_metrics, timestamp, start_time)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            # §7: the global step is cur_nimg everywhere, so curves overlay across
            # repos, batch sizes and GPU counts. This used to be kimg here.
            global_step = cur_nimg
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
        if stats_tfevents is not None:
            _write_run_hparams(run_dir, stats_tfevents,
                               {'Metrics/combra_fid_best': float(best_fid)})
        stage('Exiting')

#----------------------------------------------------------------------------
