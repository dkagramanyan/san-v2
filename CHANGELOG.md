# Changelog

All notable changes to this fork (`san-v2`) are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Fixed
- **`tests/test_combra_contract.py` asserted combra symbols the training loop no
  longer imports.** Its `REQUIRED` list still named the eight feature / angle
  functions from before the sharded harness moved into combra, and never named
  `combra.metrics.distributed`'s `all_ranks_ok` / `distributed_metrics` /
  `gather_generated` / `precompute_reference` — the four symbols the loop actually
  depends on. That is the exact blind spot the test exists to close (combra 0.5.0
  removing three functions hid for a release the same way). It now pins
  `(module, name)` pairs for every combra import in the repo and the unguarded
  import block mirrors the loop's real imports.

- **The shard merge validated nothing about the files it read.** It opened
  every `shards/rank_NNN.h5` and scanned its `written` masks, never checking
  that the file was a `generated_images_shard` at all, that its
  `schema_version` matched, or that its writer had reached `close()` — so a
  worker that died before its first batch left a root-attr-less file to be
  treated as "zero written", and a stale shard from an earlier run in the same
  `--outdir` could be merged as if it were this run's. The merge now refuses a
  shard with the wrong `format` / `schema_version` or a `class_*` group with no
  `missing_count` attr (the closed-shard marker, §4), and `san-gen-images`
  clears `shards/rank_*.h5` before spawning workers. The written-mask
  recomputation still gates every slot; these checks fail earlier and name the
  actual cause.

- `RankH5Writer`'s docstring named a `<desc_full>/` shard directory that the
  worker never used; shards live at `<outdir>/shards/rank_<rank:03d>.h5`.

### Added
- **`bfloat16` support in the three custom CUDA ops.** `bias_act`, `upfirdn2d` and
  `filtered_lrelu` accepted only `float16`/`float32`; `bfloat16` either tripped the
  `TORCH_CHECK` in the C++ layer or fell through to the slow reference path with no
  warning. Adds the `c10::BFloat16` template instantiations across `bias_act.cu`,
  `upfirdn2d.cu` and all three `filtered_lrelu` sign-mode variants, the matching
  dispatch branches, and the dtype guard in `filtered_lrelu.py`. Verified against the
  reference implementations on sm_86: max abs error 1.6e-2 in bf16, consistent with
  its 8-bit mantissa.
- **Per-phase timing in the distributed path.** `training_loop` reports
  `accumulate_gradients` and `all_reduce` wall time every 20 batches on rank 0, which
  is what separates a slow optimiser step from a slow collective.
- **H200 diagnostic scripts** — `diagnose_performance.py`, `disable_custom_ops.py`
  and `setup_h200_env.sh`.
- **`tests/test_rank_h5.py` — the h5 schema contract is now asserted.** CPU-only,
  tiny arrays, no checkpoint: the unified shard signature
  (`format="generated_images_shard"`, `schema_version=1`, `class_names`,
  `image_shape_hwc`, `samples_per_class`, per-group `class_idx`/`class_name`,
  uint8 NHWC images, int64 seeds, bool written), the shard→merged roundtrip, the
  merge hard-fail on missing slots (including deletion of the partial merged
  file), the zero-sample-rank teardown, and the mandatory-`class_names` errors.
  Nothing in `tests/` touched the h5 output before.

### Removed
- **The agent debug logger in the custom CUDA ops.** Four files carried an NDJSON
  logger hardcoded to `/home/dgkagramanyan/.cursor/debug.log`, a path that does not
  exist here; every call opened it, raised, and was swallowed by a bare
  `except Exception: pass`. The expensive part was not the logging:
  `FilteredLReluCuda.forward` held seven **unguarded** `torch.cuda.synchronize()`
  calls used only to timestamp the debug records, so every `filtered_lrelu` call
  stalled the entire GPU pipeline. 30 `#region agent log` blocks removed, with the
  imports, globals and dead timers they fed. The opt-in timing under `training/` is
  guarded by `_DEBUG_ENABLED` / call-count and is left in place.
- **Two hardcoded environment variables in `train.py`.** A merged branch set
  `NCCL_P2P_DISABLE=1` and `TORCH_CUDA_ARCH_LIST="9.0"` for every `num_gpus > 1` run,
  under a comment reading "Potential fix". The arch pin forces the JIT to build for
  Hopper regardless of the actual device — wrong on any non-Hopper card, and
  `custom_ops._get_cuda_arch_flags()` already detects the real compute capability.
  `NCCL_P2P_DISABLE=1` disables peer-to-peer transfer, which slows multi-GPU training
  on NVLink rather than speeding it up. Both are still settable from the environment.
- **`test.py`.** A 483-line standalone CUDA check at the repo root, resurrected by a
  merge after `c56e5e9` had deleted it. `tests/test_cuda_ops.py` covers the same
  ground and pytest actually collects it.
- **`todo.md`.** Every item in it was closed, so the file said nothing a reader
  needed; the fixes are described in this changelog instead.

### Fixed
- **`Timing/eval_sec` was reported on rank 0 only, desyncing the stats reduction.**
  `training_stats.report0` registers the counter *name* on whichever rank calls it —
  it discards non-zero ranks' values, not the registration — and `Collector.update()`
  all-reduces over the registered set, which the module's own docstring says must
  match across processes. The call sat inside `if rank == 0 and combra_results is not
  None`, so rank 0 carried a name no other rank had and the reduction disagreed on
  shape from the first snapshot tick of any multi-GPU run. It is now called by every
  rank, outside the guard.

- **Loading the reference slice happened before combra's rank handshake.** A decode
  error or `MemoryError` while stacking this rank's images raised outside
  `precompute_reference`, so that rank set `combra_ref_ok = False` and walked on while
  the others were already blocked in the precompute's `all_reduce`. Ranks then also
  disagreed on `combra_ref_ok`, desyncing every later tick. The stack is now guarded
  and agreed through `all_ranks_ok` before any collective.

- **Eval failures printed on rank 0 only, hiding the rank that actually failed.**
  Both the per-tick eval and the reference precompute logged under `if rank == 0`,
  so the common case — a non-zero rank OOMing — produced a run that reported a
  failure with no error attached. Every rank now prints, tagged with its rank.

- **A rank with no work crashed `gen_images` teardown and left a broken shard.**
  With `samples_per_class < world_size` the contiguous block split assigns tail
  ranks zero samples, so `RankH5Writer._init` never runs: `close()` raised
  `KeyError` and an empty, attribute-less `rank_NNN.h5` stayed on disk for the
  merge to trip over. `close()` now deletes the empty shard (nothing was
  assigned, so nothing is missing) and the merge skips absent rank files —
  legitimate only because of that deletion; a crashed rank still surfaces
  through the unchanged merged-coverage hard-fail, which counts every slot no
  shard covered.
- **The custom CUDA ops needed `ninja` on `PATH`.** Calling the env's interpreter by
  absolute path -- what every launcher that does not `conda activate` does -- left the
  env's `bin/` off `PATH`, so the JIT build failed with "Ninja is required to load C++
  extensions" although ninja sat next to the interpreter. `custom_ops.get_plugin`
  prepends `os.path.dirname(sys.executable)` before building.
- **combra is pinned to a tag (`@v0.10.0`) instead of tracking `main`.** Unpinned, every
  fresh env resolved whatever combra `main` was that day, so the FID / CMMD / FD-DINOv2 /
  angle numbers a run is judged on could change with no signal and no record. combra
  0.8.0 also stamps `combra/version` into this run's TensorBoard HPARAMS, so the metric
  code behind a run is now recoverable from its log. Local development is unaffected --
  the env's editable combra install shadows the URL.
- **`san-eval` and `calc_metrics.py` could not be imported at all.** Two independent
  import-time faults, one hidden behind the other:
  1. `feature_networks/clip/clip.py` built the vendored CLIP text tokenizer at module
     import, and its `bpe_simple_vocab_16e6.txt.gz` is not in the repo, so every
     importer of `feature_networks` raised `FileNotFoundError` -- even `--help`. The
     tokenizer is now built on first use. Nothing in this fork calls `tokenize()`:
     only the image tower is used, via `clip.load(...)[0].visual`.
  2. `dill` is used by `metrics/metric_utils.py` to load pickled feature detectors but
     was never declared. Now in `dependencies`.
- **Console scripts are now covered by a packaging test** (`tests/test_entry_points.py`).
  It launches every entry point declared in `[project.scripts]` with `--help` from a
  temp cwd, which is the only way to see this class of bug: pytest runs with the repo
  root on `sys.path`, so an in-repo test passes while the installed script is broken.
  Confirmed to fail against the pre-fix packaging before being kept.
- **`stats.jsonl` rows are built by a testable function**, and a new
  `tests/test_stats_contract.py` feeds a real row to `combra.metrics.load_fid_by_kimg`.
  The reader was only ever tested against a synthetic flat row, so nothing checked the
  producer.
- **The §7 logging contract is now asserted** (`tests/test_logging_contract.py`).
  Thirteen scalar keys had drifted across the four repos; nothing failed because
  nothing checked. See below for this repo's share.

### Changed
- **`class_names` is now mandatory for h5 generation.** The `class_names` root
  attr was stamped only when the checkpoint metadata happened to carry names, so
  a name-less checkpoint silently produced h5 files outside the label contract
  (§5). `gen_images` in hdf5 mode now raises `ValueError` before generation
  starts, and `RankH5Writer` / the shard merge refuse a `None` `class_names`
  instead of conditionally skipping the attr. `--save-mode dir` is unaffected.
- **One `_get_cuda_arch_flags()` instead of two.** Merging the CUDA branch left two
  definitions of it and injected `-gencode` flags into the same compile twice. The
  surviving version detects the device's real compute capability; the discarded one
  hardcoded `sm_80 + sm_90` and fell back to `major, minor = 9, 0  # Default to Hopper`
  when detection threw.
- **Four branches deleted upstream in January were merged back into `main`**, which
  is now the only branch on the remote: `10-very-long-loading-and-training`,
  `cursor/background-process-setup-ea47`, `cursor/missing-task-description-25a5` and
  `cursor/missing-task-description-4760`.
- **The sharded eval harness moved into combra** (`combra.metrics.distributed`). This
  repo kept only what is model-specific: producing a shard of generated images and the
  float->uint8 denormalisation. The four private copies had drifted three ways --
  `all_gather` vs `gather`, a failure flag or none, and a different
  `precompute_reference` signature in each.
- **The combra startup check is `self_test(image_metrics=True, strict=True, images=...)`.**
  A missing CLIP download previously surfaced only as a whole run logging `nan`.
- **Hyperparameters reach TensorBoard.** The resolved config is read back from
  `training_options.json` at the end of training and written to the HPARAMS tab with
  the run's final `Metrics/combra_fid_best`, so runs are comparable by configuration
  and not only by curve shape. Nothing logged them before.
- **§7 keys:** the TensorBoard global step was kimg; it is now `cur_nimg`, as the
  contract has always specified, so curves overlay across repos and GPU counts.
  `Timing/eval_sec` and `LearningRate/G` / `LearningRate/D` are now logged;
  `Timing/total_hours` / `total_days` (this repo only, just `total_sec` rescaled) are
  dropped. The sample grid is now also logged to TensorBoard as `Fakes` -- this repo is
  the reference implementation for the grid, and was the only one where it never
  reached TensorBoard. The event file finally carries the run name as a
  `filename_suffix`, which is what makes a tfevents file copied into `wc_cv/ml/`
  identifiable.
- **A failed reference precompute now disables the metrics** instead of leaving every
  tick to call the eval with `combra_ref=None` and raise inside its own `try`.

- **The combra contract test fed a unimodal sample to a bimodal-fit metric.**
  `test_angle_metrics_run_on_pooled_angles` drew two near-identical normals
  (mu 120 and 126), so the second Gaussian had no mode to sit on. combra now
  reports that as `nan` rather than dividing by the phantom, which turned the
  assertion red. The fixture is now genuinely bimodal (a 70/30 mixture at
  100 deg and 240 deg), which is what a WC-Co vertex-angle distribution
  actually looks like.
- **`scipy.linalg.sqrtm(..., disp=False)` raises under SciPy >= 1.18**, which
  removed the `disp` parameter. Fixed in `metrics/frechet_inception_distance.py`. Calling `sqrtm(X)` without `disp` returns
  the matrix alone on every SciPy version, so the fix is version-agnostic. This
  surfaced when the environment moved to SciPy 1.18 (see below); before that the
  call would have failed at runtime the moment anyone upgraded.

### Changed
- **The conda environment is now `san-v2`** (Python 3.12, torch 2.13+cu130,
  numpy 2.5, SciPy 1.18), rebuilt alongside the previous `san` env rather
  than replacing it. `requires-python` has said `>=3.12` since the v2 convention
  landed, but the working env was still 3.11 — so `pip install -e .` could not
  succeed, which is why the console scripts were missing and combra was absent.
  README and `sh/` launch scripts point at the new name.
- **CI installs combra and arms the contract test.** `tests/test_combra_contract.py`
  is entirely `skipif(not combra_installed)`, and no CI job installed combra, so the
  file could go green by doing nothing. CI now installs combra when a `COMBRA_TOKEN`
  secret is present and sets `COMBRA_REQUIRED=1`; a new always-on test fails if
  combra is missing under that flag.

## [0.3.0] — 2026-08-18

Repairs the combra integration and closes the remaining v2-convention gaps.

### Fixed
- **combra metrics were silently disabled.** The eval path imported
  `angle_density_metrics_from_pooled`, `fid_from_features` and
  `fd_dinov2_from_features`, which combra removed in 0.5.0, plus `combra_smoke_test`,
  which it renamed to `self_test`. The `except` around the per-tick eval swallowed the
  resulting `ImportError` and printed "metric evaluation failed", so
  `--combra-metrics true` produced no metrics at all. Now imports
  `frechet_from_features` (one helper for both Fréchet metrics) and `self_test`;
  combra ≥ 0.7.0 restores `angle_density_metrics_from_pooled`.
- **`[combra]` installed a combra with no metric backends.** The extra pulled bare
  `combra`; since combra 0.5.0 the torch / `pytorch-fid` / `open-clip-torch` stack is
  behind `combra[metrics]`, so FID / CMMD / FD-DINOv2 would have returned `nan` even
  after the import fix. Now `combra[metrics] @ git+…`.
- **A one-rank failure could deadlock the run.** The reference precompute went
  straight into `all_gather`, so a rank that failed to extract (absent CLIP weights,
  OOM) left the survivors blocked in a collective that never completed. A rank-uniform
  success handshake now runs between the local extraction and the first gather.
- **Stale metric rows.** `stats_metrics` persisted across ticks while combra only ran
  at snapshot ticks, so every intermediate row re-emitted the previous evaluation's
  values at a new step — turning the curves into step functions and letting post-hoc
  snapshot selection resolve to a kimg that was never evaluated. The row is cleared
  each tick.
- README pointed at a `requirements.txt` that does not exist, and showed
  `dataset_tool.py --source=…` without the `convert` subcommand the click group needs.
- **`distutils` import broke `dnnlib` on Python 3.12+.** `dnnlib/util.py` imported
  `strtobool` from `distutils`, removed from the standard library in 3.12 — the floor
  this release moves to — and available only through setuptools' own deprecated shim.
  Replaced with a local `_strtobool`.
- **`pkg_resources` import broke the custom ops on Python 3.12+.**
  `torch_utils/ops/conv2d_gradfix.py` and `grid_sample_gradfix.py` imported
  `parse_version` from `pkg_resources`, which setuptools no longer ships, so
  `import training.training_loop` failed outright on the Python floor this release
  moves to. Replaced with a small `torch_utils.misc.parse_version`.

### Changed
- **Metric keys lost the literal `10k`.** `Metrics/combra_fid10k` was emitted whatever
  `--num-fid-samples` said, so any chart built from it was mislabelled. Keys are now
  bare — `Metrics/combra_fid`, `combra_cmmd`, `combra_fd_dinov2` — and the count is
  logged once as `Metrics/combra_num_fid_samples`. combra's `load_fid_by_kimg` reads
  the bare key and still falls back to the old one for archived runs.
- **`--combra-ref-count 0` now means "the whole reference set"**, matching the other
  three repos. It was `IntRange(min=1)` with default `None` here, so the same launch
  script failed on this repo alone.
- **Angle-extraction workers scale with the rank count** (`cpu_count // gpus`, capped
  at 32). Every rank asking for `min(32, cpu_count)` oversubscribed an 8-GPU node
  eightfold.
- `requires-python` raised to **3.12** to match combra.

### Added
- `--grad-accum` (default 1), completing the shared training CLI — the other three
  repos already had it. Total batch = `batch-gpu × gpus × grad-accum`.
- `Metrics/combra_fid_best`, the running best FID, in `stats.jsonl` and TensorBoard.
- `tests/test_combra_contract.py` — asserts every combra symbol this repo imports
  actually exists. CPU-only, no GPU/dataset/network, so it runs in every CI job. This
  is the check whose absence let the breakage above survive a whole release.

## [0.2.0] — 2026-07-17

Adoption of the v2 model API convention (wc_cv `models_api_proposal`, §12). **Breaking.**

### Added
- **Console scripts** `san-train`, `san-gen-images`, `san-eval`, `san-prepare-data`
  (`pip install -e .`).
- **Self-describing inference snapshots** `san-snapshot-<kimg:06d>-inference.pt` — EMA-only
  `.pt` **state dicts** carrying `{n_classes, resolution, class_names, cur_nimg}` metadata.
  Written **atomically** (tmp + `os.replace`) every snapshot tick **and always at the last
  tick**, pruned to `--snapshot-keep-last` (default 3, `0` = keep all).
- **Training flags** `--precision {fp32,fp16,bf16}`, `--tf32`, `--bench`, `--num-fid-samples`,
  `--combra-ref-count`, `--snapshot-keep-last`.
- **Generation**: `--classes` accepts names or indices, validated against the checkpoint;
  unified h5 signature (`format="generated_images_shard"`, `schema_version=1`); `class_names`
  stamped into h5 / `classes.json`; the merge **hard-fails** on incomplete shards.
- **Dataset**: `dataset_tool` derives labels alphabetically, writes `class_names`, errors on
  missing labels, converts grayscale→RGB at build time; `san-prepare-data convert` click group
  with the shared transform set (incl. `center-crop-dhariwal`).
- combra metrics mirrored into `stats.jsonl`; startup combra self-test; normalize/denormalize
  round-trip assert; DDP weight-consistency check before saving; `tests/test_smoke.py`; ruff CI.

### Changed
- `--fp32`→`--precision`, `--nobench`→`--bench`; progressive flags to kebab-case
  (`--up-factor`, `--path-stem`, `--syn-layers`, `--head-layers`, `--cls-weight`);
  `--mirror` is now a loader-level stochastic flip (was dataset x-flip doubling).
- Run-dir name drops the dataset basename: `<cfg>-gpus<G>-batch<B>[-desc]`; a fresh id is
  always allocated. combra install via the `[combra]` extra (`git+https`).
- combra eval denorm fixed (`+128`→`+127.5`), so reals and fakes cross the uint8 boundary
  identically; class count read from `class_names`, not `max(label)+1`.

### Removed
- **Resume / restart machinery** (`--resume`, `--restart_every`, exit-code-3, rolling
  `network-snapshot-latest.pt`, `best_model.pkl`/`best_nimg.txt`), `--save-weights-only`,
  `--save-inference-only`, the native `--metrics` registry from training, Hydra
  (`train_hydra.py`, `configs/`, `hydra-core`), `requirements.txt`, pickled-module saving
  (`legacy.py`), the GUI deps, and dead files (`networks_stylegan3.py`,
  `dataset_tool_for_imagenet.py`, `sbatch/`).

### Breaking
- Interrupted runs can no longer be resumed (size `--kimg` to fit the job's time limit).
- Old `.pkl` artifacts are not readable (check out a pre-0.2.0 commit for those); combra
  metric values shift (denorm fix); commands using removed/renamed flags fail.

## Pre-release development — 2026-07-16

> Kept for history. This predates the 0.2.0 release below and was previously
> mislabelled `[Unreleased]`, which put shipped work under a heading implying it
> was pending. Much of it — `--resume`, `best_model.pkl`,
> `--save-inference-only`, `network-snapshot-latest.pt` — was **removed** by the
> v2 convention, so read it as a record of what changed then, not as current
> behaviour.

### Changed
- **Production `sbatch/train_*.sbatch` now pass `--save-inference-only 0`** (was `True`).
  A prod run keeps only the rolling `network-snapshot-latest.pt` resume checkpoint and the
  best-FID `best_model.pkl` — no per-tick `G_ema` history is accumulated. Set
  `--save-inference-only 1` to restore the per-tick inference snapshots.
- **Full resume checkpoint renamed** `network-snapshot.pkl` → **`network-snapshot-latest.pt`**
  (`torch_utils/misc.get_ckpt_path`). Behaviour is unchanged — it is still a single file
  overwritten in place every snapshot tick (never accumulates) and still carries the
  `G`/`D`/`G_ema` networks + resume `progress` (no optimizer state, as before). The
  per-tick `network-snapshot-<kimg>-inference.pkl` history snapshots serve as the
  accumulating record; the `-latest.pt` file is purely for `--resume`.
  **Migration:** existing runs must rename their on-disk `network-snapshot.pkl` to
  `network-snapshot-latest.pt` to keep auto-resuming.
- **`timm` unpinned to 1.x** (`timm>=1.0.0`, was `timm==0.4.12`) in `requirements.txt`
  and `pyproject.toml`. Updated feature-network model ids for the new timm registry:
  `tf_efficientnet_b0_ns` → `tf_efficientnet_b0.ns_jft_in1k` (`feature_networks/constants.py`);
  `vit_deit_base_patch16_384` → `deit_base_patch16_384`,
  `vit_deit_base_distilled_patch16_384` → `deit_base_distilled_patch16_384`,
  `vit_base_resnet50_384` → `vit_base_r50_s16_384` (`feature_networks/vit.py`, legacy
  DPT-ViT helpers not on the default projected-discriminator path). All other deps
  already used `>=` (latest-compatible); `glob` is stdlib.
  **⚠ Breaking:** checkpoints that embed the projected discriminator's timm feature
  networks (`best_model.pkl` stems used as `--path_stem`, and `network-snapshot-latest.pt`)
  were saved under timm 0.4.12 and will **not** unpickle under timm 1.x — the progressive
  16²→1024² stem chain must be regenerated from scratch. Inference-only `G_ema` snapshots
  are unaffected. This migration was **not** runtime-tested (timm 1.x unavailable in the
  authoring env); verify with `tests/test_san_modules.py` on the cluster before a long run.

## Pre-release development — 2026-06-25

> Kept for history. This predates the 0.2.0 release below and was previously
> mislabelled `[Unreleased]`, which put shipped work under a heading implying it
> was pending. Much of it — `--resume`, `best_model.pkl`,
> `--save-inference-only`, `network-snapshot-latest.pt` — was **removed** by the
> v2 convention, so read it as a record of what changed then, not as current
> behaviour.

### Added
- **`--save-inference-only` training flag** — writes a tiny `network-snapshot-<kimg>-inference.pkl`
  containing only `G_ema` (no discriminator, no resume state) each snapshot tick; the
  smallest artifact, intended for `gen_images.py` / `calc_metrics.py`.
  (`train.py`, `training/training_loop.py`)
- **`--combra-metrics` training flag** (default `true`) — computes the combra
  generative-quality metrics each snapshot tick, **independent of `--metrics`** (you can
  now use `--metrics none` and still get combra, or vice-versa).
- **combra image-feature metrics** (`fid`, `cmmd`, `fd_dinov2`) are now computed during
  training (the loop passes `image_metrics=True`), in addition to the angle-density
  metrics. All are logged to TensorBoard under `Metrics/combra_*`.
- **Startup warning** when `--combra-metrics=true` but the `combra` package is not
  installed (instead of a silent skip).
- **Pretrained-model downloader** — running `python tests/test_san_modules.py` directly
  pre-fetches every weight training + the combra metrics need (timm backbones,
  InceptionV3, CLIP, DINOv2). `download_models.sh` does the same via pure
  `wget`/`curl`/`git` into the right caches, for offline compute nodes.
- **Generate sbatch scripts** for 512×512 and 1024×1024.

### Changed
- **sbatch scripts** are now self-contained and submittable from the `sbatch/` folder
  (each resolves the repo root), target **2× H200**, and queue on the **`rocky`
  partition** (no reservation). They load **no system CUDA module** — the custom ops
  build against the conda toolkit via `CUDA_HOME=$CONDA_PREFIX`. Train scripts pass
  `--save-inference-only True`.
- **Hydra entry point** — `train_hydra.py` now derives all option defaults from the
  `train.py` click CLI (single source of truth), so `configs/config.yaml` only declares
  the required fields and new flags propagate automatically.
- **`legacy.load_network_pkl`** mirrors `G_ema` onto `G` when a pickle has only `G_ema`,
  so inference-only snapshots load through the existing pipeline.
- **Install** — PyTorch wheels bumped to the CUDA 13.2 index (`cu132`); install the CUDA
  compiler (`cuda-nvcc`) from conda since no system CUDA module is loaded. `timm==0.4.12`
  is required (newer timm cannot unpickle the trained `best_model.pkl` stems); re-pin it
  after installing `requirements.txt` (combra's `open-clip-torch` otherwise pulls a newer
  timm).
- **Tests** — `test.py` moved to `tests/test_cuda_ops.py` (skips cleanly under CPU-only
  CI; still runnable as a script).
- **Docs** — README rewritten as an install → test → train → generate guide; combra
  `san_v2.md` doc kept in sync.

### Removed
- FFHQ leftovers from upstream: `scripts/train_ffhq16.sh`,
  `configs/experiment/ffhq16_stem.yaml`, `configs/experiment/ffhq32_superres.yaml`.
- Dead dependencies from `requirements.txt`: `imgui`, `glfw`, `pyopengl`,
  `imageio-ffmpeg`, and the pip `ninja` (ninja is installed from conda).
- `scripts/run_tests.sh` (use `python -m pytest tests/ -v`).
- `generate_64x64` sbatch script.

### Related (combra submodule, separate repo)
- CMMD now uses the `ViT-L-14-336-quickgelu` CLIP variant to match the `openai` weights
  (silences the QuickGELU activation-mismatch warning).
