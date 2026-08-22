#!/usr/bin/env python3
"""Self-bootstrap the MLX reference environment for the repo's Python scripts.

Every script that needs `mlx` / `mlx_lm` / `mlx_vlm` (the `bench.py` MLX side
and the `mlx_*.py` truth generators) runs its own interpreter from the repo
venv instead of assuming any pre-installed env. A bare
`python3 scripts/bench.py ...` works on any machine: this module creates the
venv and installs the pinned requirements on first use.

The venv lives at `~/.cache/lisa-rs/mlxenv` by default; override with the
`LISA_MLX_VENV` env var. Nothing here depends on `/tmp/mlxenv` or the caller's
Python environment.
"""

import os
import shutil
import subprocess
import sys

SCRIPTS = os.path.dirname(os.path.abspath(__file__))  # this directory
REQUIREMENTS = os.path.join(SCRIPTS, "requirements-mlx.txt")

# Pinned so historical reruns stay reproducible (matches the env the numbers
# in AGENTS.md were measured with).
# Requirements are installed from requirements-mlx.txt at bootstrap time.


def venv_root() -> str:
    return os.environ.get("LISA_MLX_VENV") or os.path.expanduser(
        "~/.cache/lisa-rs/mlxenv"
    )


def venv_python() -> str:
    return os.path.join(venv_root(), "bin", "python")


def _find_python() -> str:
    """A modern Python (>=3.12) to seed the venv: mlx wheels for recent
    versions only publish for 3.12+. Prefer 3.14 (the version AGENTS.md numbers
    were measured with), falling back to whatever python3 the caller has."""
    candidates = [
        "/opt/homebrew/bin/python3.14",
        "/opt/homebrew/bin/python3.13",
        "/opt/homebrew/bin/python3.12",
        "/usr/local/bin/python3.14",
        "/usr/local/bin/python3.13",
        "/usr/local/bin/python3.12",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    base = shutil.which("python3") or sys.executable
    return base


def _probe(py: str) -> bool:
    try:
        r = subprocess.run(
            [py, "-c", "import mlx.core as mx; import mlx_lm; import mlx_vlm"],
            capture_output=True,
        )
        return r.returncode == 0
    except FileNotFoundError:
        return False


def ensure() -> str:
    """Return the venv python, creating + installing it if missing."""
    py = venv_python()
    if os.path.exists(py) and _probe(py):
        return py
    if not os.path.exists(py):
        os.makedirs(venv_root(), exist_ok=True)
        # Seed the venv with a modern interpreter (`-m venv` selects it, not
        # the python that happens to be running this script).
        base = _find_python()
        subprocess.check_call([base, "-m", "venv", venv_root()])
    subprocess.check_call([py, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([py, "-m", "pip", "install", "-r", REQUIREMENTS])
    if not _probe(py):
        raise SystemExit(f"mlx venv setup failed at {venv_root()}")
    return py


def reexec() -> None:
    """If the current interpreter is not the repo mlx venv, re-run this script
    under it. Call at the top of every script that imports mlx before the
    heavy imports."""
    if os.path.realpath(sys.executable) == os.path.realpath(venv_python()):
        return
    py = ensure()
    os.execv(py, [py, *sys.argv])


def run(script: str, args: list[str]) -> int:
    """Bare `python3 scripts/mlx_env.py script.py [args...]` runner."""
    py = ensure()
    return subprocess.call([py, script, *args])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python3 scripts/mlx_env.py <script.py> [args...]")
    sys.exit(run(sys.argv[1], sys.argv[2:]))