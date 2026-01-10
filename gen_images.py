import os
import tempfile
from pathlib import Path
from typing import List, Optional

import PIL.Image
import click
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm

import legacy
import dnnlib
from torch_utils import gen_utils
from gen_images import parse_range


def _sync(world_size: int) -> None:
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.barrier()


def _ensure_run_dir(run_dir: Optional[Path], cfg: dict, world_size: int, rank: int) -> Path:
    if run_dir is None and rank == 0:
        run_dir = Path(gen_utils.make_run_dir(cfg['outdir'], cfg['desc_full']))
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        dir_holder = [str(run_dir) if rank == 0 else '']
        dist.broadcast_object_list(dir_holder, src=0)
        run_dir = Path(dir_holder[0])
    if run_dir is None:
        raise RuntimeError('Failed to create run directory for sample generation.')
    return run_dir


def _init_distributed(rank: int, world_size: int, init_method: Optional[str]) -> torch.device:
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend='nccl',
            init_method=init_method or 'env://',
            rank=rank,
            world_size=world_size,
        )

    device_id = int(os.environ.get('LOCAL_RANK', rank % max(1, torch.cuda.device_count())))
    torch.cuda.set_device(device_id)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    return torch.device('cuda', device_id)


def _generate_for_rank(rank: int, world_size: int, cfg: dict, run_dir: Optional[Path], init_method: Optional[str]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA device is required for image generation.')

    device = _init_distributed(rank, world_size, init_method)
    run_dir = _ensure_run_dir(run_dir, cfg, world_size, rank)

    if rank == 0:
        print(f'Loading network from "{cfg["network_pkl"]}"...')
    with torch.inference_mode():
        with dnnlib.util.open_url(cfg['network_pkl']) as f:
            G = legacy.load_network_pkl(f)['G_ema'].eval().requires_grad_(False).to(device)

        _sync(world_size)
        class_iter = tqdm(cfg['classes'], disable=rank != 0)
        for class_idx in class_iter:
            class_dir = run_dir / f'class_{class_idx}'
            if rank == 0:
                class_dir.mkdir(parents=True, exist_ok=True)
            _sync(world_size)

            image_indices = list(range(rank, cfg['samples_per_class'], world_size))
            for start in range(0, len(image_indices), cfg['batch_gpu']):
                batch_indices = image_indices[start:start + cfg['batch_gpu']]
                if not batch_indices:
                    continue
                seeds = [
                    cfg['seed'] + class_idx * cfg['samples_per_class'] + idx
                    for idx in batch_indices
                ]

                w = gen_utils.get_w_from_seed(
                    G,
                    batch_sz=len(batch_indices),
                    device=device,
                    truncation_psi=cfg['truncation_psi'],
                    seed=None,
                    centroids_path=cfg['centroids_path'],
                    class_idx=class_idx,
                    seeds=seeds,
                )
                images = gen_utils.w_to_img(G, w, to_np=True)

                for img, img_idx in zip(images, batch_indices):
                    PIL.Image.fromarray(img, 'RGB').save(class_dir / f"{class_idx}_{img_idx}.png")
    _sync(world_size)
    if rank == 0:
        print(f'Finished writing samples to "{run_dir}".')


def _spawn_workers(num_gpus: int, cfg: dict, run_dir: Path) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        init_file = os.path.join(temp_dir, '.torch_distributed_init')
        init_method = f'file://{init_file}'
        mp.set_start_method('spawn', force=True)
        mp.spawn(
            _generate_for_rank,
            args=(num_gpus, cfg, run_dir, init_method),
            nprocs=num_gpus,
            join=True,
        )


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--trunc', 'truncation_psi', help='Truncation psi', type=float, default=1.0, show_default=True)
@click.option('--seed', help='Random seed base', type=int, default=42, show_default=True)
@click.option('--centroids-path', type=str, help='Pass path to precomputed centroids to enable multimodal truncation')
@click.option('--classes', type=parse_range, help='List of classes (e.g., \'0,1,4-6\')', required=True)
@click.option('--samples-per-class', help='Samples per class.', type=int, default=4, show_default=True)
@click.option('--batch-gpu', help='Samples per pass, adapt to fit on GPU', type=int, default=32, show_default=True)
@click.option('--gpus', help='Number of GPUs to use (defaults to all available)', type=click.IntRange(min=1))
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--desc', help='String to include in result dir name', metavar='STR', type=str)
def generate_samplesheet(
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
):
    os.makedirs(outdir, exist_ok=True)
    desc_full = f'{Path(network_pkl).stem}_trunc_{truncation_psi}'
    if desc is not None:
        desc_full += f'-{desc}'

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

    env_world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if env_world_size > 1:
        rank = int(os.environ.get('RANK', '0'))
        _generate_for_rank(rank, env_world_size, cfg, run_dir=None, init_method='env://')
        return

    num_gpus = gpus or torch.cuda.device_count() or 1
    run_dir = Path(gen_utils.make_run_dir(outdir, desc_full))
    if num_gpus == 1:
        _generate_for_rank(rank=0, world_size=1, cfg=cfg, run_dir=run_dir, init_method=None)
    else:
        _spawn_workers(num_gpus, cfg, run_dir)


if __name__ == "__main__":
    generate_samplesheet()
