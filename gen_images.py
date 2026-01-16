#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Tuple

import click
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import h5py
import PIL.Image

import legacy
import dnnlib
from torch_utils import gen_utils


# ---------------------------
# Helpers
# ---------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    if isinstance(s, list):
        return s
    out: List[int] = []
    for part in str(s).split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def _slurm_cpu_count_fallback() -> int:
    for k in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        v = os.environ.get(k)
        if v:
            try:
                n = int(v)
                if n > 0:
                    return n
            except ValueError:
                pass
    return os.cpu_count() or 1


# ---------------------------
# HDF5 writer process (rank0 only)
#   - Single writer owns h5py.File handle => no corruption
#   - Incremental writes during generation
#   - No memory growth: generator never stores full dataset
# ---------------------------

def _hdf5_writer_loop(
    queue: "mp.queues.JoinableQueue",
    hdf5_path: str,
    classes: List[int],
    samples_per_class: int,
    compression: str,
    chunk_images: int,
) -> None:
    """
    Receives items: (class_idx:int, sample_idxs:np.ndarray[int], seeds:np.ndarray[int64], images:np.ndarray[uint8])
    Writes into fixed-size datasets at positions sample_idxs (order independent).
    """
    hdf5_path = str(hdf5_path)
    Path(hdf5_path).parent.mkdir(parents=True, exist_ok=True)

    f = h5py.File(hdf5_path, "w")

    # Lazy init once we know image shape.
    initialized = False
    img_shape = None  # (H, W, C)

    groups: Dict[int, h5py.Group] = {}
    d_images: Dict[int, h5py.Dataset] = {}
    d_seeds: Dict[int, h5py.Dataset] = {}
    d_written: Dict[int, h5py.Dataset] = {}

    def _init_dsets(_img_shape: Tuple[int, int, int]) -> None:
        nonlocal initialized, img_shape
        img_shape = tuple(int(x) for x in _img_shape)
        H, W, C = img_shape

        f.attrs["format"] = "stylegan_generated_images"
        f.attrs["image_shape_hwc"] = img_shape
        f.attrs["samples_per_class"] = int(samples_per_class)
        f.attrs["classes"] = np.array(classes, dtype=np.int32)

        for class_idx in classes:
            g = f.create_group(f"class_{int(class_idx)}")
            g.attrs["class_idx"] = int(class_idx)
            g.attrs["samples_per_class"] = int(samples_per_class)
            g.attrs["image_shape_hwc"] = img_shape

            # Fixed-size datasets (no resizing = fewer leaks/fragmentation).
            # Write-at-index is safe even if batches arrive out of order.
            chunks0 = max(1, min(int(chunk_images), int(samples_per_class)))
            dimg = g.create_dataset(
                "images",
                shape=(samples_per_class, H, W, C),
                dtype=np.uint8,
                chunks=(chunks0, H, W, C),
                compression=compression if compression else None,
                shuffle=True if compression else False,
            )
            dseed = g.create_dataset(
                "seeds",
                shape=(samples_per_class,),
                dtype=np.int64,
                chunks=(max(1, min(chunks0 * 4, samples_per_class)),),
                compression=None,
            )
            dw = g.create_dataset(
                "written",
                shape=(samples_per_class,),
                dtype=np.bool_,
                chunks=(max(1, min(chunks0 * 4, samples_per_class)),),
                compression=None,
            )
            dw[:] = False

            groups[class_idx] = g
            d_images[class_idx] = dimg
            d_seeds[class_idx] = dseed
            d_written[class_idx] = dw

        initialized = True
        f.flush()

    try:
        while True:
            item = queue.get()
            try:
                if item is None:
                    # Sentinel: finish and exit.
                    return

                class_idx, sample_idxs, seeds, images = item
                class_idx = int(class_idx)

                # images: (B, H, W, C) uint8
                if not initialized:
                    if images.ndim != 4:
                        raise RuntimeError(f"Writer expected images with ndim=4, got {images.ndim}")
                    _init_dsets(tuple(images.shape[1:]))

                # Validate shape consistency
                if tuple(images.shape[1:]) != tuple(img_shape):
                    raise RuntimeError(
                        f"Image shape mismatch: got {tuple(images.shape[1:])}, expected {tuple(img_shape)}"
                    )

                # Ensure numpy dtypes (avoid accidental object arrays)
                sample_idxs = np.asarray(sample_idxs, dtype=np.int64)
                seeds = np.asarray(seeds, dtype=np.int64)
                images = np.asarray(images, dtype=np.uint8)

                if sample_idxs.size != images.shape[0] or seeds.size != images.shape[0]:
                    raise RuntimeError(
                        f"Batch size mismatch: idxs={sample_idxs.size}, seeds={seeds.size}, images={images.shape[0]}"
                    )

                # Sort indices for slightly better IO locality
                order = np.argsort(sample_idxs)
                sample_idxs = sample_idxs[order]
                seeds = seeds[order]
                images = images[order]

                # Bounds check
                if sample_idxs.min(initial=0) < 0 or sample_idxs.max(initial=-1) >= samples_per_class:
                    raise RuntimeError(
                        f"sample_idxs out of range for class {class_idx}: "
                        f"min={int(sample_idxs.min())}, max={int(sample_idxs.max())}, "
                        f"samples_per_class={samples_per_class}"
                    )

                # Write into fixed positions
                d_images[class_idx][sample_idxs, :, :, :] = images
                d_seeds[class_idx][sample_idxs] = seeds
                d_written[class_idx][sample_idxs] = True

                # Flush frequently so data is on disk DURING generation
                f.flush()

            finally:
                queue.task_done()

    finally:
        # Final flush + quick integrity info
        try:
            for class_idx in classes:
                if class_idx in d_written:
                    written = int(np.count_nonzero(d_written[class_idx][:]))
                    groups[class_idx].attrs["written_count"] = written
                    groups[class_idx].attrs["missing_count"] = int(samples_per_class - written)
            f.flush()
        except Exception:
            pass
        f.close()


# ---------------------------
# Directory save (per-rank, no contention)
# ---------------------------

def _save_images_to_dir(
    outdir: Path,
    class_idx: int,
    sample_idxs: np.ndarray,
    seeds: np.ndarray,
    images: np.ndarray,
    fmt: str = "png",
) -> None:
    class_dir = outdir / f"class_{int(class_idx)}"
    class_dir.mkdir(parents=True, exist_ok=True)

    sample_idxs = np.asarray(sample_idxs)
    seeds = np.asarray(seeds)
    images = np.asarray(images, dtype=np.uint8)

    # One file per sample_idx (stable, deterministic)
    for i in range(images.shape[0]):
        sid = int(sample_idxs[i])
        seed = int(seeds[i])
        img = images[i]  # HWC uint8
        im = PIL.Image.fromarray(img)
        # Include both class, sample index and seed in filename
        fn = class_dir / f"idx_{sid:06d}_seed_{seed}.{fmt}"
        im.save(str(fn))


# ---------------------------
# Generation worker
# ---------------------------

def _generate_worker(
    rank: int,
    world_size: int,
    cfg: dict,
    init_method: Optional[str],
    output_hdf5_path: Optional[Path],
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for image generation.")

    # Avoid CPU thread oversubscription across ranks
    torch.set_num_threads(1)

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method=init_method or "env://",
            rank=rank,
            world_size=world_size,
        )

    # Map ranks to GPUs (respect CUDA_VISIBLE_DEVICES if set)
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        available_gpus = [int(x) for x in cuda_visible.split(",") if x.strip() != ""]
        device_id = available_gpus[rank % len(available_gpus)]
    else:
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            device_id = int(local_rank)
        else:
            device_id = rank % max(1, torch.cuda.device_count())

    torch.cuda.set_device(device_id)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda", device_id)

    classes: List[int] = list(cfg["classes"])
    samples_per_class: int = int(cfg["samples_per_class"])
    batch_gpu: int = int(cfg["batch_gpu"])
    truncation_psi: float = float(cfg["truncation_psi"])
    centroids_path = cfg.get("centroids_path", None)
    seed_base: int = int(cfg["seed"])

    save_mode: str = cfg["save_mode"]  # "hdf5" or "dir"
    outdir: Path = Path(cfg["outdir"])
    image_format: str = cfg["image_format"]
    hdf5_compression: str = cfg["hdf5_compression"]
    hdf5_chunk_images: int = int(cfg["hdf5_chunk_images"])
    writer_queue_max: int = int(cfg["writer_queue_max"])

    # Rank0 decides output_hdf5_path
    if save_mode == "hdf5":
        if output_hdf5_path is None and rank == 0:
            run_dir = Path(gen_utils.make_run_dir(cfg["outdir"], cfg["desc_full"]))
            output_hdf5_path = run_dir / f"{cfg['desc_full']}.h5"
        if world_size > 1:
            holder = [str(output_hdf5_path) if rank == 0 else ""]
            dist.broadcast_object_list(holder, src=0)
            output_hdf5_path = Path(holder[0])

    # Global progress bar (rank0)
    total_images = int(len(classes) * samples_per_class)
    pbar = None
    if rank == 0:
        pbar = tqdm(
            total=total_images,
            unit="img",
            dynamic_ncols=True,
            desc="Generating",
            smoothing=0.05,
        )

    # Start writer process on rank0 for HDF5 mode
    writer_proc = None
    writer_queue = None
    if save_mode == "hdf5" and rank == 0:
        ctx = mp.get_context("spawn")
        writer_queue = ctx.JoinableQueue(maxsize=max(1, writer_queue_max))
        writer_proc = ctx.Process(
            target=_hdf5_writer_loop,
            args=(
                writer_queue,
                str(output_hdf5_path),
                classes,
                samples_per_class,
                hdf5_compression,
                hdf5_chunk_images,
            ),
            daemon=True,
        )
        writer_proc.start()

    # Load network
    if rank == 0:
        print(f'Loading network from "{cfg["network_pkl"]}"...')

    with torch.inference_mode():
        with dnnlib.util.open_url(cfg["network_pkl"]) as f:
            G = legacy.load_network_pkl(f)["G_ema"].eval().requires_grad_(False).to(device)

        # progress sync batching (distributed)
        local_done_since_sync = 0
        sync_every = max(256, batch_gpu * 4)

        # Generate class by class
        for class_idx in classes:
            # Each rank generates its shard of indices
            img_ids = list(range(rank, samples_per_class, world_size))

            for start in range(0, len(img_ids), batch_gpu):
                batch_ids = img_ids[start:start + batch_gpu]
                if not batch_ids:
                    continue

                # Deterministic per-sample seed
                seeds = [seed_base + int(class_idx) * samples_per_class + int(idx) for idx in batch_ids]

                w = gen_utils.get_w_from_seed(
                    G,
                    batch_sz=len(batch_ids),
                    device=device,
                    truncation_psi=truncation_psi,
                    seed=None,
                    centroids_path=centroids_path,
                    class_idx=class_idx,
                    seeds=seeds,
                )
                images_list = gen_utils.w_to_img(G, w, to_np=True)  # list of HWC uint8

                # Convert to contiguous numpy batch for transfer/write
                images_np = np.stack(images_list, axis=0).astype(np.uint8, copy=False)  # (B,H,W,C)
                sample_idxs_np = np.asarray(batch_ids, dtype=np.int64)
                seeds_np = np.asarray(seeds, dtype=np.int64)

                # Free GPU-side refs ASAP to avoid leaks
                del w
                # (Optional) helps reduce fragmentation; keep light
                # torch.cuda.empty_cache()

                # Save DURING generation
                if save_mode == "dir":
                    # Safe to write per-rank directly (unique filenames), no gather needed
                    _save_images_to_dir(outdir, class_idx, sample_idxs_np, seeds_np, images_np, fmt=image_format)
                else:
                    # save_mode == "hdf5": gather to rank0 and enqueue to writer
                    payload = (int(class_idx), sample_idxs_np, seeds_np, images_np)

                    if world_size == 1:
                        # Single GPU: directly enqueue
                        writer_queue.put(payload)
                    else:
                        # Gather per-batch payloads from all ranks to rank0
                        gathered: List[Any] = [None for _ in range(world_size)] if rank == 0 else None
                        dist.gather_object(payload, object_gather_list=gathered, dst=0)

                        if rank == 0:
                            # Enqueue each rank's payload (skip empty)
                            for item in gathered:
                                if item is None:
                                    continue
                                cidx, sidxs, sds, imgs = item
                                if imgs is None or len(sidxs) == 0:
                                    continue
                                writer_queue.put(item)

                # Update progress
                local_done_since_sync += len(batch_ids)

                if world_size > 1 and local_done_since_sync >= sync_every:
                    t = torch.tensor([local_done_since_sync], device=device, dtype=torch.long)
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    if rank == 0 and pbar is not None:
                        pbar.update(int(t.item()))
                    local_done_since_sync = 0
                elif world_size == 1 and rank == 0 and pbar is not None:
                    pbar.update(len(batch_ids))

                # Drop CPU refs ASAP (avoid accidental accumulation)
                del images_list
                del images_np
                del sample_idxs_np
                del seeds_np

        # Final progress sync for distributed
        if world_size > 1 and local_done_since_sync > 0:
            t = torch.tensor([local_done_since_sync], device=device, dtype=torch.long)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            if rank == 0 and pbar is not None:
                pbar.update(int(t.item()))
            local_done_since_sync = 0

    if rank == 0 and pbar is not None:
        pbar.close()

    # Finalize writer (rank0)
    if save_mode == "hdf5" and rank == 0:
        # Wait until all ranks finished generating before stopping writer
        if world_size > 1:
            dist.barrier()

        # Drain queue and stop writer
        writer_queue.join()          # wait all queued items processed
        writer_queue.put(None)       # sentinel
        writer_queue.join()          # mark sentinel as done
        writer_proc.join()

        # Quick final info
        size_mb = output_hdf5_path.stat().st_size / (1024 * 1024)
        print(f'✓ HDF5 written incrementally: "{output_hdf5_path}" ({size_mb:.1f} MB)')

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


# ---------------------------
# CLI
# ---------------------------

@click.command()
@click.option("--network", "network_pkl", required=True, help="Network pickle filename")
@click.option("--trunc", "truncation_psi", type=float, default=1.0, show_default=True, help="Truncation psi")
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed base")
@click.option("--centroids-path", type=str, default=None, help="Path to precomputed centroids for multimodal truncation")
@click.option("--classes", type=parse_range, required=True, help="List of classes, e.g. '0,1,4-6'")
@click.option("--samples-per-class", type=int, default=4, show_default=True, help="Samples per class")
@click.option("--batch-gpu", type=int, default=32, show_default=True, help="Samples per GPU pass")
@click.option("--gpus", type=click.IntRange(min=1), help="GPUs to use (defaults to all)")
@click.option("--outdir", type=str, required=True, metavar="DIR", help="Where to save outputs (dir or HDF5)")
@click.option("--desc", type=str, metavar="STR", help="String to include in result filename")
@click.option("--output-hdf5", type=str, metavar="FILE", help="Output HDF5 filename (only for --save-mode hdf5)")
@click.option(
    "--save-mode",
    type=click.Choice(["hdf5", "dir"], case_sensitive=False),
    default="hdf5",
    show_default=True,
    help="Save as one big HDF5 (incremental during generation) OR as folder with images.",
)
@click.option(
    "--image-format",
    type=click.Choice(["png", "jpg", "jpeg"], case_sensitive=False),
    default="png",
    show_default=True,
    help="Image format for --save-mode dir",
)
@click.option(
    "--hdf5-compression",
    type=click.Choice(["lzf", "none"], case_sensitive=False),
    default="lzf",
    show_default=True,
    help="HDF5 compression (lzf is fast). Use 'none' for raw speed + bigger file.",
)
@click.option(
    "--hdf5-chunk-images",
    type=int,
    default=256,
    show_default=True,
    help="Chunk length (images) along first dimension for HDF5 datasets.",
)
@click.option(
    "--writer-queue-max",
    type=int,
    default=8,
    show_default=True,
    help="Max pending batches buffered for HDF5 writer (bounds RAM; prevents leaks).",
)
def generate_images(
    network_pkl: str,
    truncation_psi: float,
    seed: int,
    centroids_path: Optional[str],
    classes: List[int],
    samples_per_class: int,
    batch_gpu: int,
    gpus: Optional[int],
    outdir: str,
    desc: Optional[str],
    output_hdf5: Optional[str],
    save_mode: str,
    image_format: str,
    hdf5_compression: str,
    hdf5_chunk_images: int,
    writer_queue_max: int,
):
    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    desc_full = f"{Path(network_pkl).stem}_trunc_{truncation_psi}" + (f"-{desc}" if desc else "")

    # Normalize compression arg
    hdf5_compression_norm = None if hdf5_compression.lower() == "none" else hdf5_compression.lower()

    # Determine output HDF5 path (only if needed)
    output_path: Optional[Path] = None
    if save_mode.lower() == "hdf5":
        if output_hdf5:
            output_path = outdir_p / output_hdf5
        else:
            run_dir = Path(gen_utils.make_run_dir(str(outdir_p), desc_full))
            output_path = run_dir / f"{desc_full}.h5"

    cfg = dict(
        network_pkl=network_pkl,
        truncation_psi=truncation_psi,
        seed=seed,
        centroids_path=centroids_path,
        classes=classes,
        samples_per_class=samples_per_class,
        batch_gpu=batch_gpu,
        outdir=str(outdir_p),
        desc_full=desc_full,
        save_mode=save_mode.lower(),
        image_format=image_format.lower(),
        hdf5_compression=hdf5_compression_norm,
        hdf5_chunk_images=hdf5_chunk_images,
        writer_queue_max=writer_queue_max,
    )

    # If launched with torchrun/srun with WORLD_SIZE, respect it
    env_world = int(os.environ.get("WORLD_SIZE", "1"))
    if env_world > 1:
        rank = int(os.environ.get("RANK", "0"))
        _generate_worker(rank, env_world, cfg, "env://", output_path)
        return

    # Otherwise local spawn to match --gpus
    num_gpus = gpus or (torch.cuda.device_count() or 1)
    if num_gpus == 1:
        _generate_worker(0, 1, cfg, None, output_path)
        return

    with mp.get_context("spawn").Manager():  # ensures spawn context is usable
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            init_file = os.path.join(temp_dir, ".torch_distributed_init")
            init_method = f"file://{init_file}"

            mp.set_start_method("spawn", force=True)
            mp.spawn(
                _generate_worker,
                args=(num_gpus, cfg, init_method, output_path),
                nprocs=num_gpus,
                join=True,
            )


if __name__ == "__main__":
    generate_images()
