"""
SC-CLM V5B2 — ReactionT5 + LoRA（服务器/ Linux 友好入口）
---------------------------------------------------------
与 V5B1 共用内核 `train_v5b1.py`，独立输出目录与日志。
默认使用仓库内 `models/ReactionT5v2-forward`（若存在），否则 HF `sagawa/ReactionT5v2-forward`；
可用 `V5B_REACTION_MODEL_PATH` 覆盖。

默认：完整 LoRA、micro_batch=1、grad_accum=8、ReactionT5 侧 FP16 开启（与 V5B1 一致）。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))


def _maybe_reexec_in_repo_venv() -> None:
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

    os.environ["V5_LORA_BRANCH_DISPLAY"] = "V5B2"
    os.environ["V5B1_MODEL_FAMILY"] = "reactiont5"
    os.environ.setdefault("V5B1_OUTPUT_DIR", "results/checkpoints/v5b2")
    os.environ.setdefault("V5B1_LOG_FILE", "logs/v5b2/train.log")
    os.environ.setdefault("V5_STATUS_FILE", str(_REPO / "logs" / "v5b2" / "_status.txt"))
    os.environ.setdefault("V5B1_MICRO_BATCH", "1")
    os.environ.setdefault("V5B1_GRAD_ACCUM", "8")
    os.environ.setdefault("V5B1_FORCE_FP16", "1")

    target = Path(__file__).resolve().with_name("train_v5b1.py")
    rc = subprocess.call([sys.executable, str(target)], env=os.environ.copy())
    raise SystemExit(rc)
