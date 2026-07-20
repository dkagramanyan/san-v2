---
name: run-san-v2
description: Run, smoke-test, or screenshot san-v2 (StyleSAN-XL GAN) — launch the train/gen_images/dataset_tool CLIs, run the CPU test suite, build a dataset zip, and generate an image on GPU. Use when asked to run, start, build, test, or verify san-v2 works.
---

# Run san-v2

`san-v2` is a StyleSAN-XL (StyleGAN3 + Projected-GAN + SAN) fork for generating
WC-Co microstructure SEM images. Its "app" is four click CLIs — `train.py`,
`gen_images.py`, `dataset_tool.py`, `calc_metrics.py` — over the StyleGAN3 model
stack with JIT-compiled custom CUDA ops. Real training/generation needs multi-GB
dataset zips and trained checkpoints that live on the H200 cluster; **neither is
in a clean checkout.**

The driver, [.claude/skills/run-san-v2/driver.py](.claude/skills/run-san-v2/driver.py),
exercises every surface that IS reachable on a bare workstation + one GPU: the
CLI contracts, the CPU test suite, a real `dataset_tool convert`, and a GPU
generator forward pass (a tiny model built in-process — no checkpoint required).

> Paths below are relative to the repo root (`san-v2/`).

## Run (agent path) — the driver

Always use the **`san` conda env's** interpreter (base conda has no torch):

```bash
/home/david/anaconda3/envs/san/bin/python .claude/skills/run-san-v2/driver.py all
```

That runs all four checks and prints a PASS/FAIL summary (exit 0 = all green).
Artifacts (the dataset zip, the generated PNG) land in a temp `--workdir` whose
path it prints. Run one check at a time with `cli`, `test`, `dataset`, or `gen`;
pin the output location with `--workdir DIR`:

```bash
/home/david/anaconda3/envs/san/bin/python .claude/skills/run-san-v2/driver.py gen --workdir /tmp/sanout
# -> /tmp/sanout/gen_forward.png  (3 class samples from an untrained tiny G = noise; proves the pipeline runs)
```

Expected `all` output (~1 min; the GPU CUDA-op tests JIT-compile on first run):

```
[ OK ] train.py --help  (flags present)
[ OK ] gen_images.py --help  (flags present)
[ OK ] dataset_tool.py --help  (flags present)
[ OK ] pytest tests/  -> 23 passed, 7 warnings in 51.80s
[ OK ] dataset_tool convert -> ds_64x64.zip (12 imgs, class_names=['Ultra_Co11', 'Ultra_Co25', 'Ultra_Co6_2'])
[ OK ] gen forward on NVIDIA GeForce RTX 3090 (11.4M params) -> .../gen_forward.png
```

## Prerequisites

The `san` conda env already exists here and has everything
(torch 2.9.1+cu128, torchvision, ninja, timm 0.4.12, click, imageio, h5py, …).
Nothing to install. Verify it:

```bash
/home/david/anaconda3/envs/san/bin/python -c "import torch,timm; print(torch.__version__, torch.cuda.is_available(), timm.__version__)"
# 2.9.1 True 0.4.12
```

If recreating the env on the cluster, follow the README §1 (torch cu13 wheels +
`conda install -c nvidia cuda-nvcc` + `pip install ninja` from conda +
`pip install -e .` + re-pin `timm==0.4.12` last).

## Run individual CLIs by hand

Run via `python <script>.py` (the package isn't pip-installed, so the
`san-train`/`san-gen-images` console scripts don't exist here). Activate the env
first so `ninja` is on `PATH` for the custom ops:

```bash
conda activate san      # or: export PATH=/home/david/anaconda3/envs/san/bin:$PATH
python dataset_tool.py convert --source=<img_dir> --dest=out_64x64.zip --resolution=64x64
python train.py --help
python gen_images.py --help
```

Real `train.py` / `gen_images.py` runs are **not** doable here — they need
dataset zips, a `--network` checkpoint, and the `in_embeddings/` weights that
only exist on the cluster. The driver's `gen` check is the in-container stand-in.

## Gotchas

- **Run with the env interpreter, and get `ninja` on `PATH`.** The custom CUDA
  ops (`bias_act`, `filtered_lrelu`, …) JIT-link via `ninja`, which torch finds
  through `PATH`. Calling `/…/envs/san/bin/python` by absolute path does *not*
  put the env's `bin/` on `PATH`, so `ninja` is invisible and the ops fail with
  *"Ninja is required to load C++ extensions."* The driver fixes this itself
  (prepends its interpreter's dir to `PATH`); by hand, `conda activate san` first.
- **`nvcc` is absent in this container — the ops still work off a warm cache.**
  There is no `nvcc` on this machine, but the custom ops load a prebuilt `.so`
  from `~/.cache/torch_extensions/py311_cu128/…-rtx-3090/`. A *cold* build (new
  machine or cleared cache) would fail — that's why the cluster install needs
  `conda install -c nvidia cuda-nvcc`.
- **`calc_metrics.py` fails at import** — it opens
  `feature_networks/clip/bpe_simple_vocab_16e6.txt.gz`, which isn't in the repo,
  at module load. So `calc_metrics.py --help` errors and the driver skips it.
  (`train.py`, `gen_images.py`, `dataset_tool.py` all import fine.)
- **README drift.** The README says `pip install -r requirements.txt`, but there
  is **no** `requirements.txt` — deps live in `pyproject.toml` (`pip install -e .`).
  It also shows `python dataset_tool.py --source …`, but the CLI is now a click
  group: `python dataset_tool.py convert --source …`.
- **Generation needs class embeddings.** The `MappingNetwork` loads
  `in_embeddings/tf_efficientnet_lite0.pkl` (a pickled `{'embed': nn.Embedding}`)
  at construction, overridable via `$SAN_EMBED`. That file is absent here, so the
  driver's `gen` check writes a tiny synthetic 3-class table and points
  `SAN_EMBED` at it.

## Troubleshooting

- `ModuleNotFoundError: No module named 'torch'` → you used base conda. Use
  `/home/david/anaconda3/envs/san/bin/python`.
- `RuntimeError: Ninja is required to load C++ extensions` → `ninja` not on
  `PATH`; `conda activate san` (the driver handles this automatically).
- `gen -> no CUDA device` → no GPU visible; the `cli`, `test`, and `dataset`
  checks still run without one.
- pytest's first run is slow (~50s) because the GPU CUDA-op tests compile; reruns
  hit the cache and finish in seconds.
