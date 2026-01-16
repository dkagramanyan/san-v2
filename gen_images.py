import os
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Union

import PIL.Image
import click
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm

import legacy
import dnnlib
from torch_utils import gen_utils


def parse_range(s: Union[str, List]) -> List[int]:
    if isinstance(s, list):
        return s
    out: List[int] = []
    for part in str(s).split(','):
        if '-' in part:
            lo, hi = part.split('-')
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def _archive_run_dir(run_dir: Path, delete_folder: bool = True) -> Path:
    """
    Create a .tar.gz archive using ALL CPUs via pigz, then optionally delete the folder.
    Archive is created next to the folder: <run_dir>.tar.gz
    """
    run_dir = run_dir.resolve()
    parent = run_dir.parent
    name = run_dir.name
    archive_path = run_dir.with_suffix(run_dir.suffix + ".tar.gz") if run_dir.suffix else run_dir.with_name(run_dir.name + ".tar.gz")

    pigz = shutil.which("pigz")
    if pigz is None:
        raise RuntimeError(
            'Requested multi-CPU archiving, but "pigz" was not found in PATH. '
            'Install pigz (e.g., `sudo dnf install -y pigz` / `sudo apt-get install -y pigz`) and retry.'
        )

    nproc = os.cpu_count() or 1

    # tar -cf - -C <parent> <name> | pigz -p <nproc> > <archive_path>
    tar_cmd = ["tar", "-cf", "-", "-C", str(parent), name]
    pigz_cmd = [pigz, "-p", str(nproc)]

    with open(archive_path, "wb") as f_out:
        tar_p = subprocess.Popen(tar_cmd, stdout=subprocess.PIPE)
        try:
            pigz_p = subprocess.Popen(pigz_cmd, stdin=tar_p.stdout, stdout=f_out)
            assert tar_p.stdout is not None
            tar_p.stdout.close()  # allow tar to receive SIGPIPE if pigz exits early
            pigz_rc = pigz_p.wait()
            tar_rc = tar_p.wait()
        finally:
            # Ensure processes are not left running on exceptions.
            if tar_p.poll() is None:
                tar_p.kill()
            # pigz_p is only defined if tar succeeded up to Popen; keep simple:
            # (in practice, exceptions above are rare; the code path is safe enough)

            pass

    if tar_rc != 0:
        raise RuntimeError(f"tar failed with exit code {tar_rc}")
    if pigz_rc != 0:
        raise RuntimeError(f"pigz failed with exit code {pigz_rc}")

    print(f'Archived results to "{archive_path}". (pigz -p {nproc})')

    if delete_folder:
        shutil.rmtree(run_dir)
        print(f'Deleted folder "{run_dir}".')

    return archive_path


def _generate_worker(
    rank: int,
    world_size: int,
    cfg: dict,
    init_method: Optional[str],
    run_dir: Optional[Path],
    archive: bool = False,
    delete_folder: bool = True,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for image generation.")

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method=init_method or "env://",
            rank=rank,
            world_size=world_size,
        )

    device_id = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count())))
    torch.cuda.set_device(device_id)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda", device_id)

    if run_dir is None and rank == 0:
        run_dir = Path(gen_utils.make_run_dir(cfg["outdir"], cfg["desc_full"]))
    if world_size > 1:
        holder = [str(run_dir) if rank == 0 else ""]
        dist.broadcast_object_list(holder, src=0)
        run_dir = Path(holder[0])
        dist.barrier()

    if rank == 0:
        print(f'Loading network from "{cfg["network_pkl"]}"...')

    with torch.inference_mode():
        with dnnlib.util.open_url(cfg["network_pkl"]) as f:
            G = legacy.load_network_pkl(f)["G_ema"].eval().requires_grad_(False).to(device)

        for class_idx in tqdm(cfg["classes"], disable=rank != 0):
            class_dir = run_dir / f"class_{class_idx}"
            if rank == 0:
                class_dir.mkdir(parents=True, exist_ok=True)
            if world_size > 1:
                dist.barrier()

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
                for img, img_idx in zip(images, batch_ids):
                    PIL.Image.fromarray(img, "RGB").save(class_dir / f"{class_idx}_{img_idx}.png")

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        print(f'Finished writing samples to "{run_dir}".')
        if archive:
            _archive_run_dir(run_dir, delete_folder=delete_folder)

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
@click.option("--outdir", type=str, required=True, metavar="DIR", help="Where to save images")
@click.option("--desc", type=str, metavar="STR", help="String to include in result dir name")
@click.option("--archive", is_flag=True, help="If set, archive the result folder and print the archive name")
@click.option(
    "--delete-folder/--no-delete-folder",
    default=True,
    show_default=True,
    help="Delete the result folder after archiving (only applies if --archive is set)",
)
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
    archive: bool,
    delete_folder: bool,
):
    os.makedirs(outdir, exist_ok=True)
    desc_full = f"{Path(network_pkl).stem}_trunc_{truncation_psi}" + (f"-{desc}" if desc else "")
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

    env_world = int(os.environ.get("WORLD_SIZE", "1"))
    if env_world > 1:
        rank = int(os.environ.get("RANK", "0"))
        _generate_worker(rank, env_world, cfg, "env://", None, archive=archive, delete_folder=delete_folder)
        return

    num_gpus = gpus or torch.cuda.device_count() or 1
    if num_gpus == 1:
        run_dir = Path(gen_utils.make_run_dir(outdir, desc_full))
        _generate_worker(
            rank=0,
            world_size=1,
            cfg=cfg,
            init_method=None,
            run_dir=run_dir,
            archive=archive,
            delete_folder=delete_folder,
        )
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        init_file = os.path.join(temp_dir, ".torch_distributed_init")
        init_method = f"file://{init_file}"
        run_dir = Path(gen_utils.make_run_dir(outdir, desc_full))
        mp.set_start_method("spawn", force=True)
        mp.spawn(
            _generate_worker,
            args=(num_gpus, cfg, init_method, run_dir, archive, delete_folder),
            nprocs=num_gpus,
            join=True,
        )


if __name__ == "__main__":
    generate_images()
