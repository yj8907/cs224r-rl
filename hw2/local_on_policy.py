#!/usr/bin/env python3
"""Local runner for CS224R HW2 on-policy training (converted from Modal)."""

import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

# --- paths (assumes this file sits next to train_on_policy.py, cfgs/, demos/) ---
REPO_ROOT = Path(__file__).resolve().parent
CFGS_DIR = REPO_ROOT / "cfgs"
CONFIG_NAME = "on_policy_config"
RUN_DIR = REPO_ROOT / "on_policy_run"          # Hydra output dir

# --- mujoco / rendering; override MUJOCO_PY_MUJOCO_PATH or MUJOCO_GL via env ---
# MUJOCO_ROOT = os.environ.get(
#     "MUJOCO_PY_MUJOCO_PATH",
#     os.path.expanduser("~/.mujoco/mujoco210"),
# )


def build_env():
    env = dict(os.environ)
    # env["MUJOCO_PY_MUJOCO_PATH"] = MUJOCO_ROOT
    # env.setdefault("MUJOCO_GL", "osmesa")       # switch to "egl" if you have a GPU + display
    # env.setdefault("PYOPENGL_PLATFORM", "osmesa")
    # if sys.platform.startswith("linux"):
    #     ld = f"{MUJOCO_ROOT}/bin:/usr/lib/x86_64-linux-gnu"
    #     existing = env.get("LD_LIBRARY_PATH")
    #     env["LD_LIBRARY_PATH"] = ld + (os.pathsep + existing if existing else "")
    return env


def patch_config(src_cfgs: Path, dst_cfgs: Path) -> Path:
    """Copy cfgs/ to a temp dir and patch for a local, single-process run."""
    shutil.copytree(src_cfgs, dst_cfgs, dirs_exist_ok=True)
    cfg_path = dst_cfgs / f"{CONFIG_NAME}.yaml"

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    cfg["use_tb"] = False
    cfg["defaults"] = [
        d for d in cfg.get("defaults", [])
        if d != "override hydra/launcher: submitit_local"
    ]
    if "hydra" in cfg:
        cfg["hydra"].pop("launcher", None)
        cfg["hydra"].pop("sweep", None)
        cfg["hydra"]["run"] = {"dir": str(RUN_DIR)}

    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return cfg_path


def main():
    if not (REPO_ROOT / "train_on_policy.py").exists():
        sys.exit(f"train_on_policy.py not found in {REPO_ROOT}")
    # if not Path(MUJOCO_ROOT).exists():
    #     sys.exit(f"MuJoCo not found at {MUJOCO_ROOT} (set MUJOCO_PY_MUJOCO_PATH)")

    RUN_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        patched_cfgs = Path(tmp) / "cfgs"
        patch_config(CFGS_DIR, patched_cfgs)

        result = subprocess.run(
            [
                sys.executable, "-u", "train_on_policy.py",
                "--config-path", str(patched_cfgs),
                "--config-name", CONFIG_NAME,
            ],
            cwd=str(REPO_ROOT),
            env=build_env(),
        )

    if result.returncode != 0:
        raise SystemExit(f"Training failed with exit code {result.returncode}")


if __name__ == "__main__":
    main()