"""
SC-CLM V5A1 Calibration Script
------------------------------
MolT5-small + LoRA low-VRAM calibration wrapper.
Reuses the shared LoRA trainer in train_v5b1.py.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Default profile for V5A1 (MolT5 + LoRA).
# Force override to avoid leaking previous shell env from V5B1 runs.
os.environ["V5B1_MODEL_FAMILY"] = "molt5"
os.environ.setdefault("V5B1_OUTPUT_DIR", "results/checkpoints/v5a1_calib")
os.environ.setdefault("V5B1_LOG_FILE", "logs/v5a1/train.log")
os.environ.setdefault("V5B1_MICRO_BATCH", "1")
os.environ.setdefault("V5B1_GRAD_ACCUM", "8")
# max_steps: omit to use train_v5b1 default (-1 = full epoch LoRA). Set V5B1_MAX_STEPS=300 for calibration.
# MolT5 LoRA uses fp32 for stability by default; can be overridden externally.
os.environ.setdefault("V5B1_FORCE_FP16", "0")


if __name__ == "__main__":
    _repo = Path(__file__).resolve().parents[3]
    _vpy = _repo / "venv" / "Scripts" / "python.exe"
    # IDE/Cursor 常用系统 Python 起脚本，会与 venv 各跑一套 LoRA、抢写同一 OUTPUT_DIR；强制回到仓库 venv。
    if _vpy.is_file() and Path(sys.executable).resolve() != _vpy.resolve():
        raise SystemExit(
            subprocess.call([str(_vpy), str(Path(__file__).resolve()), *sys.argv[1:]], env=os.environ.copy())
        )
    target = Path(__file__).resolve().with_name("train_v5b1.py")
    rc = subprocess.call([sys.executable, str(target)], env=os.environ.copy())
    raise SystemExit(rc)
