# TODO — san-v2 issues found while building the run skill (2026-07-21)

Real problems surfaced while driving the app in a clean container. All are
actionable; none block the driver, which works around them.

- [ ] **`calc_metrics.py` fails at import.** It opens
  `feature_networks/clip/bpe_simple_vocab_16e6.txt.gz` at module load, but that
  file isn't in the repo, so even `python calc_metrics.py --help` crashes with
  `FileNotFoundError`. Ship the vocab file (it's the standard CLIP BPE table) or
  make the load lazy so `--help` works.

- [ ] **README references a non-existent `requirements.txt`.** README §1 says
  `pip install -r requirements.txt`, but there is no such file — deps live in
  `pyproject.toml`. Fix to `pip install -e .` (or `.[dev]` / `.[combra]`).

- [ ] **README `dataset_tool.py` command is stale.** README §3 shows
  `python dataset_tool.py --source=… --dest=… --resolution=…`, but the CLI is now
  a click group: the flags live under the `convert` subcommand
  (`python dataset_tool.py convert --source=… --dest=… --resolution=…`).

- [ ] **`ninja` must be on `PATH` for the custom CUDA ops** (`bias_act`,
  `filtered_lrelu`, …). Calling the env's python by absolute path (not
  `conda activate`) leaves the env `bin/` off `PATH`, so the ops fail with
  "Ninja is required to load C++ extensions." Consider documenting this, or
  resolving `ninja` from the interpreter's own dir. (The run skill's driver
  already prepends `os.path.dirname(sys.executable)` to `PATH`.)
