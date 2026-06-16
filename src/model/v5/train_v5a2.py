"""
SC-CLM V5A2 — MolT5 + LoRA（服务器/ Linux 友好入口）
----------------------------------------------------
与 V5A1 共用内核 `train_v5b1.py`，独立输出目录与日志，避免与 V5A1/V5B 混写。

默认：完整 LoRA（V5B1_MAX_STEPS=-1）、micro_batch=1、grad_accum=8、MolT5 关闭强制 FP16。
环境变量与 V5A1/V5B1 相同（均以 V5B1_* 为前缀），见 CLAUDE.md。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))


def _maybe_reexec_in_repo_venv() -> None:
    """若存在仓库 venv 且当前解释器不是该 venv，则切到 venv 再跑（Windows / Linux）。"""
    for cand in (
        _REPO / "venv" / "Scripts" / "python.exe",
        _REPO / "venv" / "bin" / "python",
        _REPO / "venv" / "bin" / "python3",
    ):
        if not cand.is_file():
            continue
        try:
            if Path(sys.executable).resolve() == cand.resolve():
                return
        except OSError:
            return
        raise SystemExit(
            subprocess.call(
                [str(cand), str(Path(__file__).resolve()), *sys.argv[1:]],
                env=os.environ.copy(),
            )
        )


if __name__ == "__main__":
    _maybe_reexec_in_repo_venv()

    os.environ["V5_LORA_BRANCH_DISPLAY"] = "V5A2"
    os.environ["V5B1_MODEL_FAMILY"] = "molt5"
    os.environ.setdefault("V5B1_OUTPUT_DIR", "results/checkpoints/v5a2")
    os.environ.setdefault("V5B1_LOG_FILE", "logs/v5a2/train.log")
    os.environ.setdefault("V5_STATUS_FILE", str(_REPO / "logs" / "v5a2" / "_status.txt"))
    os.environ.setdefault("V5B1_MICRO_BATCH", "1")
    os.environ.setdefault("V5B1_GRAD_ACCUM", "8")
    os.environ.setdefault("V5B1_FORCE_FP16", "0")

    target = Path(__file__).resolve().with_name("train_v5b1.py")
    rc = subprocess.call([sys.executable, str(target)], env=os.environ.copy())
    raise SystemExit(rc)
