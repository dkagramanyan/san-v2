import os
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

import click
import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
import h5py

import legacy
import dnnlib
from torch_utils import gen_utils


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
    # Prefer cpus-per-task; fallback to cpus on node; fallback to OS count.
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


def _save_to_hdf5(hdf5_path: Path, images_data: Dict[int, List[np.ndarray]], max_workers: int = None) -> None:
    """
    Save images to HDF5 format with groups per class.
    Uses multiple threads for efficient writing.
    """
    if max_workers is None:
        max_workers = _slurm_cpu_count_fallback()

    with h5py.File(hdf5_path, 'w') as f:
        for class_idx, images in images_data.items():
            if not images:
                continue

            # Create group for this class
            group = f.create_group(f'class_{class_idx}')

            # Get image shape (assume all images have same shape)
            img_shape = images[0].shape
            num_images = len(images)

            # Create dataset for images
            dset = group.create_dataset(
                'images',
                shape=(num_images, *img_shape),
                dtype=np.uint8,
                chunks=(min(100, num_images), *img_shape),  # Reasonable chunk size
                compression='gzip',
                compression_opts=6
            )

            # Save images using multiple threads for large datasets
            if num_images <= 1000:  # For smaller datasets, save directly
                for i, img in enumerate(images):
                    dset[i] = img
            else:  # For larger datasets, use threading
                def _write_chunk(start_idx: int, end_idx: int) -> None:
                    for i in range(start_idx, min(end_idx, num_images)):
                        dset[i] = images[i]

                chunk_size = max(1, num_images // max_workers)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    for start in range(0, num_images, chunk_size):
                        end = min(start + chunk_size, num_images)
                        futures.append(executor.submit(_write_chunk, start, end))

                    for future in futures:
                        future.result()

            # Store metadata
            group.attrs['class_idx'] = class_idx
            group.attrs['num_images'] = num_images
            group.attrs['image_shape'] = img_shape


def _generate_worker(
    rank: int,
    world_size: int,
    cfg: dict,
    init_method: Optional[str],
    output_path: Optional[Path],
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for image generation.")

    # Avoid CPU thread oversubscription across ranks (common on HPC)
    torch.set_num_threads(1)

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method=init_method or "env://",
            rank=rank,
            world_size=world_size,
        )

    # Map ranks to GPUs uniformly
    # Check if CUDA_VISIBLE_DEVICES is set and respect it
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        available_gpus = [int(x) for x in cuda_visible.split(",")]
        device_id = available_gpus[rank % len(available_gpus)]
    else:
        # Fallback to LOCAL_RANK or round-robin assignment
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

    # Create/broadcast output_path once (no per-class barriers)
    if output_path is None and rank == 0:
        run_dir = Path(gen_utils.make_run_dir(cfg["outdir"], cfg["desc_full"]))
        output_path = run_dir / f"{cfg['desc_full']}.h5"

    if world_size > 1:
        holder = [str(output_path) if rank == 0 else ""]
        dist.broadcast_object_list(holder, src=0)
        output_path = Path(holder[0])

    if rank == 0:
        print(f'Loading network from "{cfg["network_pkl"]}"...')

    # Total images for a global progress bar on rank0:
    # total = len(classes) * samples_per_class (independent of world_size)
    total_images = int(len(cfg["classes"]) * cfg["samples_per_class"])

    # Rank0 progress bar updated via distributed all-reduce
    pbar = None
    if rank == 0:
        pbar = tqdm(
            total=total_images,
            unit="img",
            dynamic_ncols=True,
            desc="Generating",
            smoothing=0.05,
        )

    # Store images in memory per class
    local_images: Dict[int, List[np.ndarray]] = {class_idx: [] for class_idx in cfg["classes"]}

    # local counter; we report progress in chunks to keep comm overhead low
    local_done_since_sync = 0
    sync_every = max(256, cfg["batch_gpu"] * 4)  # tune: fewer all-reduces

    with torch.inference_mode():
        with dnnlib.util.open_url(cfg["network_pkl"]) as f:
            G = legacy.load_network_pkl(f)["G_ema"].eval().requires_grad_(False).to(device)

        for class_idx in cfg["classes"]:
            # Shard image indices across ranks
            img_ids = list(range(rank, cfg["samples_per_class"], world_size))

            for start in range(0, len(img_ids), cfg["batch_gpu"]):
                batch_ids = img_ids[start : start + cfg["batch_gpu"]]
                if not batch_ids:
                    continue

                seeds = [cfg["seed"] + class_idx * cfg["samples_per_class"] + idx for idx in batch_ids]

                w = gen_utils.get_w_from_seed(
                    G,
                    batch_sz=len(batch_ids),
                    device=device,
                    truncation_psi=cfg["truncation_psi"],
                    seed=None,
                    centroids_path=cfg["centroids_path"],
                    class_idx=class_idx,
                    seeds=seeds,
                )
                images = gen_utils.w_to_img(G, w, to_np=True)

                # Store images in memory
                local_images[class_idx].extend(images)

                # Progress accounting: count generated images
                local_done_since_sync += len(batch_ids)

                # Periodically update global progress bar (rank0) with an all-reduce
                if world_size > 1 and local_done_since_sync >= sync_every:
                    t = torch.tensor([local_done_since_sync], device=device, dtype=torch.long)
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    if rank == 0 and pbar is not None:
                        pbar.update(int(t.item()))
                    local_done_since_sync = 0

                elif world_size == 1 and rank == 0 and pbar is not None:
                    # Single GPU: update immediately
                    pbar.update(len(batch_ids))

    # Final progress sync for distributed mode (flush remaining local_done_since_sync)
    if world_size > 1:
        if local_done_since_sync > 0:
            t = torch.tensor([local_done_since_sync], device=device, dtype=torch.long)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            if rank == 0 and pbar is not None:
                pbar.update(int(t.item()))
            local_done_since_sync = 0

    if rank == 0 and pbar is not None:
        pbar.close()

    # Gather all images to rank 0 and save to HDF5
    if world_size > 1:
        # Gather image counts per class from all ranks
        local_counts = {class_idx: len(images) for class_idx, images in local_images.items()}
        all_counts = [{} for _ in range(world_size)]
        dist.all_gather_object(all_counts, local_counts)

        if rank == 0:
            # Calculate total images per class
            total_images_per_class = {}
            for counts in all_counts:
                for class_idx, count in counts.items():
                    total_images_per_class[class_idx] = total_images_per_class.get(class_idx, 0) + count

            # Prepare to receive images
            gathered_images: Dict[int, List[np.ndarray]] = {class_idx: [] for class_idx in cfg["classes"]}

            # Gather images class by class to manage memory
            for class_idx in cfg["classes"]:
                if total_images_per_class.get(class_idx, 0) == 0:
                    continue

                # Collect local images for this class
                local_class_images = local_images[class_idx]

                # Gather all images for this class from all ranks
                all_class_images = [[] for _ in range(world_size)]
                dist.all_gather_object(all_class_images, local_class_images)

                # Flatten the gathered images
                for rank_images in all_class_images:
                    gathered_images[class_idx].extend(rank_images)

            # Save to HDF5
            print(f'Saving images to "{output_path}"...')
            _save_to_hdf5(output_path, gathered_images)
            print(f'Finished saving {sum(len(imgs) for imgs in gathered_images.values())} images to HDF5.')
        else:
            # Non-zero ranks: participate in gathering
            for class_idx in cfg["classes"]:
                local_class_images = local_images[class_idx]
                all_class_images = [[] for _ in range(world_size)]
                dist.all_gather_object(all_class_images, local_class_images)

    else:
        # Single GPU: save directly
        print(f'Saving images to "{output_path}"...')
        _save_to_hdf5(output_path, local_images)
        print(f'Finished saving {sum(len(imgs) for imgs in local_images.values())} images to HDF5.')

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


@click.command()
@click.option("--network", "network_pkl", required=True, help="Network pickle filename")
@click.option("--trunc", "truncation_psi", type=float, default=1.0, show_default=True, help="Truncation psi")
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed base")
@click.option("--centroids-path", type=str, help="Path to precomputed centroids for multimodal truncation")
@click.option("--classes", type=parse_range, required=True, help="List of classes, e.g. '0,1,4-6'")
@click.option("--samples-per-class", type=int, default=4, show_default=True, help="Samples per class")
@click.option("--batch-gpu", type=int, default=32, show_default=True, help="Samples per GPU pass")
@click.option("--gpus", type=click.IntRange(min=1), help="GPUs to use (defaults to all)")
@click.option("--outdir", type=str, required=True, metavar="DIR", help="Where to save the HDF5 file")
@click.option("--desc", type=str, metavar="STR", help="String to include in result filename")
@click.option("--output-hdf5", type=str, metavar="FILE", help="Output HDF5 filename (overrides automatic naming)")
def generate_images(
    network_pkl: str,
    truncation_psi: float,
    seed: int,
    centroids_path: str,
    classes: List[int],
    samples_per_class: int,
    batch_gpu: int,
    gpus: Optional[int],
    outdir: str,
    desc: str,
    output_hdf5: Optional[str],
):
    os.makedirs(outdir, exist_ok=True)
    desc_full = f"{Path(network_pkl).stem}_trunc_{truncation_psi}" + (f"-{desc}" if desc else "")

    # Determine output HDF5 path
    if output_hdf5:
        output_path = Path(outdir) / output_hdf5
    else:
        run_dir = Path(gen_utils.make_run_dir(outdir, desc_full))
        output_path = run_dir / f"{desc_full}.h5"

    cfg = dict(
        network_pkl=network_pkl,
        truncation_psi=truncation_psi,
        seed=seed,
        centroids_path=centroids_path,
        classes=classes,
        samples_per_class=samples_per_class,
        batch_gpu=batch_gpu,
        outdir=outdir,
        desc_full=desc_full,
    )

    # If launched with srun/torchrun with WORLD_SIZE, respect it
    env_world = int(os.environ.get("WORLD_SIZE", "1"))
    if env_world > 1:
        rank = int(os.environ.get("RANK", "0"))
        _generate_worker(rank, env_world, cfg, "env://", output_path)
        return

    # Otherwise, local spawn to match --gpus
    num_gpus = gpus or torch.cuda.device_count() or 1
    if num_gpus == 1:
        _generate_worker(0, 1, cfg, None, output_path)
        return

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
