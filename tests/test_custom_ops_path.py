"""`ninja` must be findable when python is invoked by absolute path.

torch's C++ extension builder shells out to `ninja` from PATH. Calling the env's
interpreter directly (no `conda activate`) leaves the env's bin/ off PATH, so the
custom CUDA ops died with "Ninja is required to load C++ extensions" even though
ninja was installed right next to the interpreter. No GPU or compiler needed here
-- this checks the PATH repair itself.
"""

import os
import shutil
import subprocess
import sys

from torch_utils.custom_ops import _ensure_interpreter_bin_on_path


def test_adds_the_interpreter_bin_dir(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    bindir = _ensure_interpreter_bin_on_path()
    assert os.environ["PATH"].split(os.pathsep)[0] == bindir
    assert bindir == os.path.dirname(sys.executable)


def test_is_idempotent(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    _ensure_interpreter_bin_on_path()
    once = os.environ["PATH"]
    _ensure_interpreter_bin_on_path()
    assert os.environ["PATH"] == once


def test_ninja_is_resolvable_from_a_stripped_path():
    # The real failure: a stripped PATH cannot see ninja until the repair runs.
    if shutil.which("ninja", path=os.path.dirname(sys.executable)) is None:
        import pytest
        pytest.skip("ninja is not installed in this env")
    probe = (
        "import os, shutil, sys;"
        "before = shutil.which('ninja');"
        "sys.path.insert(0, os.getcwd());"
        "from torch_utils.custom_ops import _ensure_interpreter_bin_on_path as f;"
        "f();"
        "print(before, shutil.which('ninja') is not None)"
    )
    env = dict(os.environ, PATH="/usr/bin:/bin")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                         env=env, cwd=os.getcwd()).stdout.strip()
    assert out.endswith("True"), out
    assert out.startswith("None"), f"PATH was not actually stripped: {out}"
