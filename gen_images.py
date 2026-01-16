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


def _rank_to_device_id(rank: int) -> int:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        ids = [int(x) for x in cuda_visible.split(",") if x.strip() != ""]
        return ids[rank % len(ids)]
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        return int(local_rank)
    return rank % max(1, torch.cuda.device_count())


def _split_indices_block(n: int, rank: int, world_size: int) -> np.ndarray:
    """
    Contiguous block split so every rank has almost the same count.
    This avoids rank0 getting a systematically smaller/odd workload
    for small n or when n % world_size != 0.
    """
    # block partition: [start, end)
    base = n // world_size
    rem = n % world_size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    if start >= end:
        return np.empty((0,), dtype=np.int64)
    return np.arange(start, end, dtype=np.int64)


# ---------------------------
# Per-rank incremental HDF5 writer (no cross-rank sync)
# ---------------------------

class RankH5Writer:
    """
    Each rank writes its own shard file:
      <outdir>/<desc_full>/shards/rank_000.h5
    Incremental (during generation), fixed-size datasets, bounded memory.
    """
    def __init__(
        self,
        shard_path: Path,
        classes: List[int],
        samples_per_class: int,
        compression: Optional[str],
        chunk_images: int,
    ):
        self.shard_path = shard_path
        self.classes = [int(c) for c in classes]
        self.samples_per_class = int(samples_per_class)
        self.compression = compression
        self.chunk_images = int(chunk_images)

        self.f: Optional[h5py.File] = None
        self.initialized = False
        self.img_shape: Optional[Tuple[int, int, int]] = None

        self.d_images: Dict[int, h5py.Dataset] = {}
        self.d_seeds: Dict[int, h5py.Dataset] = {}
        self.d_written: Dict[int, h5py.Dataset] = {}

    def open(self):
        self.shard_path.parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(str(self.shard_path), "w")

    def _init(self, img_shape: Tuple[int, int, int]):
        assert self.f is not None
        H, W, C = [int(x) for x in img_shape]
        self.img_shape = (H, W, C)

        self.f.attrs["format"] = "stylegan_generated_images_shard"
        self.f.attrs["image_shape_hwc"] = self.img_shape
        self.f.attrs["samples_per_class"] = int(self.samples_per_class)
        self.f.attrs["classes"] = np.array(self.classes, dtype=np.int32)

        chunks0 = max(1, min(self.chunk_images, self.samples_per_class))

        for c in self.classes:
            g = self.f.create_group(f"class_{c}")
            g.attrs["class_idx"] = int(c)
            g.attrs["samples_per_class"] = int(self.samples_per_class)
            g.attrs["image_shape_hwc"] = self.img_shape

            dimg = g.create_dataset(
                "images",
                shape=(self.samples_per_class, H, W, C),
                dtype=np.uint8,
                chunks=(chunks0, H, W, C),
                compression=self.compression if self.compression else None,
                shuffle=True if self.compression else False,
            )
            dseed = g.create_dataset(
                "seeds",
                shape=(self.samples_per_class,),
                dtype=np.int64,
                chunks=(max(1, min(chunks0 * 4, self.samples_per_class)),),
                compression=None,
            )
            dw = g.create_dataset(
                "written",
                shape=(self.samples_per_class,),
                dtype=np.bool_,
                chunks=(max(1, min(chunks0 * 4, self.samples_per_class)),),
                compression=None,
            )
            dw[:] = False

            self.d_images[c] = dimg
            self.d_seeds[c] = dseed
            self.d_written[c] = dw

        self.initialized = True
        self.f.flush()

    def write_batch(self, class_idx: int, sample_idxs: np.ndarray, seeds: np.ndarray, images: np.ndarray):
        assert self.f is not None
        class_idx = int(class_idx)

        sample_idxs = np.asarray(sample_idxs, dtype=np.int64)
        seeds = np.asarray(seeds, dtype=np.int64)
        images = np.asarray(images, dtype=np.uint8)

        if not self.initialized:
            if images.ndim != 4:
                raise RuntimeError(f"Expected images (B,H,W,C), got {images.shape}")
            self._init(tuple(images.shape[1:]))

        if tuple(images.shape[1:]) != tuple(self.img_shape):
            raise RuntimeError(f"Shape mismatch: got {images.shape[1:]}, expected {self.img_shape}")

        if sample_idxs.size != images.shape[0] or seeds.size != images.shape[0]:
            raise RuntimeError("Batch mismatch idxs/seeds/images")

        # Sort by index for IO locality
        order = np.argsort(sample_idxs)
        sample_idxs = sample_idxs[order]
        seeds = seeds[order]
        images = images[order]

        # Write-at-index
        self.d_images[class_idx][sample_idxs, :, :, :] = images
        self.d_seeds[class_idx][sample_idxs] = seeds
        self.d_written[class_idx][sample_idxs] = True

        # Flush during generation (you requested “during script working”)
        self.f.flush()

    def close(self):
        if self.f is None:
            return
        # Store counts
        for c in self.classes:
            written = int(np.count_nonzero(self.d_written[c][:]))
            grp = self.f[f"class_{c}"]
            grp.attrs["written_count"] = written
            grp.attrs["missing_count"] = int(self.samples_per_class - written)
        self.f.flush()
        self.f.close()
        self.f = None


# ---------------------------
# Optional merge (rank0 after generation)
# ---------------------------

def _merge_shards_to_one_h5(
    merged_path: Path,
    shards_dir: Path,
    classes: List[int],
    samples_per_class: int,
    compression: Optional[str],
    chunk_images: int,
    world_size: int,
):
    """
    Merge per-rank shards into one big HDF5.
    Deterministic: for each class and index, take the first shard that has written=True.
    """
    merged_path.parent.mkdir(parents=True, exist_ok=True)

    # Open all shards read-only
    shard_files = [h5py.File(str(shards_dir / f"rank_{r:03d}.h5"), "r") for r in range(world_size)]
    try:
        # Determine image shape from first shard that has it
        img_shape = None
        for sf in shard_files:
            if "image_shape_hwc" in sf.attrs:
                img_shape = tuple(sf.attrs["image_shape_hwc"])
                break
        if img_shape is None:
            raise RuntimeError("No shard contains image shape metadata (did generation run?).")

        H, W, C = [int(x) for x in img_shape]

        with h5py.File(str(merged_path), "w") as out:
            out.attrs["format"] = "stylegan_generated_images"
            out.attrs["image_shape_hwc"] = (H, W, C)
            out.attrs["samples_per_class"] = int(samples_per_class)
            out.attrs["classes"] = np.array([int(c) for c in classes], dtype=np.int32)
            out.attrs["world_size"] = int(world_size)
            out.attrs["merged_from"] = str(shards_dir)

            chunks0 = max(1, min(int(chunk_images), int(samples_per_class)))

            for c in classes:
                c = int(c)
                g = out.create_group(f"class_{c}")
                g.attrs["class_idx"] = c
                g.attrs["samples_per_class"] = int(samples_per_class)
                g.attrs["image_shape_hwc"] = (H, W, C)

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

                # Merge by scanning shards
                for r, sf in enumerate(shard_files):
                    grp = sf.get(f"class_{c}", None)
                    if grp is None:
                        continue
                    wmask = np.asarray(grp["written"][:], dtype=bool)
                    if not wmask.any():
                        continue
                    # only write those not yet written
                    need = wmask & (~dw[:])
                    if not need.any():
                        continue
                    idxs = np.nonzero(need)[0]
                    dimg[idxs] = grp["images"][idxs]
                    dseed[idxs] = grp["seeds"][idxs]
                    dw[idxs] = True
                    out.flush()

                g.attrs["written_count"] = int(np.count_nonzero(dw[:]))
                g.attrs["missing_count"] = int(samples_per_class - g.attrs["written_count"])
                out.flush()
    finally:
        for sf in shard_files:
            sf.close()


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

    for i in range(images.shape[0]):
        sid = int(sample_idxs[i])
        seed = int(seeds[i])
        im = PIL.Image.fromarray(images[i])
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
    run_dir: Path,
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

    device_id = _rank_to_device_id(rank)
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
    image_format: str = cfg["image_format"]
    hdf5_compression: Optional[str] = cfg["hdf5_compression"]
    hdf5_chunk_images: int = int(cfg["hdf5_chunk_images"])

    # Progress bar: compute-only (all ranks do equal work; pbar only on rank0)
    total_images = int(len(classes) * samples_per_class)
    pbar = None
    if rank == 0:
        pbar = tqdm(total=total_images, unit="img", dynamic_ncols=True, desc="Generating", smoothing=0.05)

    # Per-rank shard writer (NO rank0 gather in the hot loop => uniform GPU utilization)
    shard_writer = None
    shards_dir = run_dir / "shards"
    if save_mode == "hdf5":
        shard_path = shards_dir / f"rank_{rank:03d}.h5"
        shard_writer = RankH5Writer(
            shard_path=shard_path,
            classes=classes,
            samples_per_class=samples_per_class,
            compression=hdf5_compression,
            chunk_images=hdf5_chunk_images,
        )
        shard_writer.open()

    if rank == 0:
        print(f'Loading network from "{cfg["network_pkl"]}"...')

    with torch.inference_mode():
        with dnnlib.util.open_url(cfg["network_pkl"]) as f:
            G = legacy.load_network_pkl(f)["G_ema"].eval().requires_grad_(False).to(device)

        local_done_since_sync = 0
        sync_every = max(256, batch_gpu * 4)

        for class_idx in classes:
            # Balanced block split (instead of stride) for stable utilization
            my_ids = _split_indices_block(samples_per_class, rank, world_size)
            if my_ids.size == 0:
                continue

            for start in range(0, my_ids.size, batch_gpu):
                batch_ids = my_ids[start:start + batch_gpu]
                if batch_ids.size == 0:
                    continue

                # Seeds per sample index (deterministic)
                seeds = seed_base + int(class_idx) * samples_per_class + batch_ids.astype(np.int64)
                seeds_list = [int(s) for s in seeds.tolist()]  # gen_utils expects python ints

                w = gen_utils.get_w_from_seed(
                    G,
                    batch_sz=int(batch_ids.size),
                    device=device,
                    truncation_psi=truncation_psi,
                    seed=None,
                    centroids_path=centroids_path,
                    class_idx=int(class_idx),
                    seeds=seeds_list,
                )
                images_list = gen_utils.w_to_img(G, w, to_np=True)  # list of HWC uint8

                images_np = np.stack(images_list, axis=0).astype(np.uint8, copy=False)
                sample_idxs_np = batch_ids.astype(np.int64, copy=False)
                seeds_np = seeds.astype(np.int64, copy=False)

                # Free GPU refs ASAP (avoid fragmentation)
                del w

                # Save DURING generation (no cross-rank sync here)
                if save_mode == "dir":
                    outdir = Path(cfg["outdir"])
                    _save_images_to_dir(outdir, int(class_idx), sample_idxs_np, seeds_np, images_np, fmt=image_format)
                else:
                    shard_writer.write_batch(int(class_idx), sample_idxs_np, seeds_np, images_np)

                # Update progress
                local_done_since_sync += int(batch_ids.size)

                if world_size > 1 and local_done_since_sync >= sync_every:
                    t = torch.tensor([local_done_since_sync], device=device, dtype=torch.long)
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    if rank == 0 and pbar is not None:
                        pbar.update(int(t.item()))
                    local_done_since_sync = 0
                elif world_size == 1 and rank == 0 and pbar is not None:
                    pbar.update(int(batch_ids.size))

                # Drop CPU refs
                del images_list, images_np, sample_idxs_np, seeds_np

        if world_size > 1 and local_done_since_sync > 0:
            t = torch.tensor([local_done_since_sync], device=device, dtype=torch.long)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            if rank == 0 and pbar is not None:
                pbar.update(int(t.item()))
            local_done_since_sync = 0

    if pbar is not None:
        pbar.close()

    if shard_writer is not None:
        shard_writer.close()

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


# ---------------------------
# CLI / launcher
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
@click.option(
    "--save-mode",
    type=click.Choice(["hdf5", "dir"], case_sensitive=False),
    default="hdf5",
    show_default=True,
)
@click.option(
    "--merge/--no-merge",
    default=True,
    show_default=True,
    help="If save-mode=hdf5: merge per-rank shards into one big HDF5 after generation.",
)
@click.option("--output-hdf5", type=str, metavar="FILE", help="Final merged HDF5 filename (only if --merge)")
@click.option(
    "--image-format",
    type=click.Choice(["png", "jpg", "jpeg"], case_sensitive=False),
    default="png",
    show_default=True,
)
@click.option(
    "--hdf5-compression",
    type=click.Choice(["lzf", "none"], case_sensitive=False),
    default="lzf",
    show_default=True,
)
@click.option(
    "--hdf5-chunk-images",
    type=int,
    default=256,
    show_default=True,
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
    save_mode: str,
    merge: bool,
    output_hdf5: Optional[str],
    image_format: str,
    hdf5_compression: str,
    hdf5_chunk_images: int,
):
    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    desc_full = f"{Path(network_pkl).stem}_trunc_{truncation_psi}" + (f"-{desc}" if desc else "")
    run_dir = Path(gen_utils.make_run_dir(str(outdir_p), desc_full))

    # Normalize compression
    compression_norm = None if hdf5_compression.lower() == "none" else hdf5_compression.lower()

    cfg = dict(
        network_pkl=network_pkl,
        truncation_psi=truncation_psi,
        seed=seed,
        centroids_path=centroids_path,
        classes=classes,
        samples_per_class=samples_per_class,
        batch_gpu=batch_gpu,
        outdir=str(run_dir if save_mode.lower() == "hdf5" else outdir_p),
        desc_full=desc_full,
        save_mode=save_mode.lower(),
        image_format=image_format.lower(),
        hdf5_compression=compression_norm,
        hdf5_chunk_images=hdf5_chunk_images,
    )

    # If launched with torchrun/srun, respect that world
    env_world = int(os.environ.get("WORLD_SIZE", "1"))
    if env_world > 1:
        rank = int(os.environ.get("RANK", "0"))
        _generate_worker(rank, env_world, cfg, "env://", run_dir)
        # Merge only on rank0
        if save_mode.lower() == "hdf5" and merge and rank == 0:
            shards_dir = run_dir / "shards"
            final_path = (run_dir / (output_hdf5 or f"{desc_full}.h5"))
            _merge_shards_to_one_h5(
                merged_path=final_path,
                shards_dir=shards_dir,
                classes=classes,
                samples_per_class=samples_per_class,
                compression=compression_norm,
                chunk_images=hdf5_chunk_images,
                world_size=env_world,
            )
            size_mb = final_path.stat().st_size / (1024 * 1024)
            print(f'✓ Merged HDF5: "{final_path}" ({size_mb:.1f} MB)')
        return

    # Otherwise local spawn
    num_gpus = gpus or (torch.cuda.device_count() or 1)
    if num_gpus == 1:
        _generate_worker(0, 1, cfg, None, run_dir)
        if save_mode.lower() == "hdf5" and merge:
            shards_dir = run_dir / "shards"
            final_path = (run_dir / (output_hdf5 or f"{desc_full}.h5"))
            _merge_shards_to_one_h5(
                merged_path=final_path,
                shards_dir=shards_dir,
                classes=classes,
                samples_per_class=samples_per_class,
                compression=compression_norm,
                chunk_images=hdf5_chunk_images,
                world_size=1,
            )
            size_mb = final_path.stat().st_size / (1024 * 1024)
            print(f'✓ Merged HDF5: "{final_path}" ({size_mb:.1f} MB)')
        return

    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        init_file = os.path.join(temp_dir, ".torch_distributed_init")
        init_method = f"file://{init_file}"

        mp.set_start_method("spawn", force=True)
        mp.spawn(
            _generate_worker,
            args=(num_gpus, cfg, init_method, run_dir),
            nprocs=num_gpus,
            join=True,
        )

    # Merge after spawn (single-process here)
    if save_mode.lower() == "hdf5" and merge:
        shards_dir = run_dir / "shards"
        final_path = (run_dir / (output_hdf5 or f"{desc_full}.h5"))
        _merge_shards_to_one_h5(
            merged_path=final_path,
            shards_dir=shards_dir,
            classes=classes,
            samples_per_class=samples_per_class,
            compression=compression_norm,
            chunk_images=hdf5_chunk_images,
            world_size=num_gpus,
        )
        size_mb = final_path.stat().st_size / (1024 * 1024)
        print(f'✓ Merged HDF5: "{final_path}" ({size_mb:.1f} MB)')


if __name__ == "__main__":
    generate_images()
