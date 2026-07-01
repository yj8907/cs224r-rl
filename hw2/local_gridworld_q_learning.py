import os
import subprocess
import sys
from pathlib import Path

# --- Environment setup that the Modal image used to provide ---------------
# These only matter if gridworld_q_learning.py (or its deps) touches MuJoCo.
# A pure gridworld Q-learning task probably doesn't need any of this, in which
# case you can delete this whole block. setdefault() means anything you've
# already exported in your shell wins.
HOME = Path.home()
MUJOCO_PATH = os.environ.get(
    "MUJOCO_PY_MUJOCO_PATH", str(HOME / ".mujoco" / "mujoco210")
)
os.environ.setdefault("MUJOCO_PY_MUJOCO_PATH", MUJOCO_PATH)
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

mujoco_bin = str(Path(MUJOCO_PATH) / "bin")
existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = (
    f"{mujoco_bin}:{existing_ld}" if existing_ld else mujoco_bin
)


def main():
    # Weights & Biases: locally this reads WANDB_API_KEY from the environment,
    # or the credentials cached by `wandb login`. (On Modal this came from the
    # `wandb-secret` Modal Secret.) Set WANDB_MODE=offline to log without
    # uploading, or WANDB_MODE=disabled to turn it off entirely.
    if not os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE") not in (
        "offline",
        "disabled",
    ):
        print(
            "Note: WANDB_API_KEY not set. Run `wandb login` (or export "
            "WANDB_API_KEY) to log to Weights & Biases, or set "
            "WANDB_MODE=offline / disabled.",
            file=sys.stderr,
        )

    script_dir = Path(__file__).resolve().parent
    result = subprocess.run(
        [sys.executable, "-u", "gridworld_q_learning.py", *sys.argv[1:]],
        cwd=script_dir,
    )

    if result.returncode != 0:
        raise SystemExit(f"Training failed with exit code {result.returncode}")


if __name__ == "__main__":
    main()