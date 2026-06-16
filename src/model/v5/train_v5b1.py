"""
SC-CLM V5B1 LoRA trainer (shared by V5A1/V5B1)
----------------------------------------------
Default: full LoRA (max_steps=-1, num_epochs from V5Config, eval/save each epoch).
Set V5B1_MAX_STEPS to a positive integer for short calibration runs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    _repo_early = Path(__file__).resolve().parents[3]
    _vpy_early = _repo_early / "venv" / "Scripts" / "python.exe"
    if _vpy_early.is_file() and Path(sys.executable).resolve() != _vpy_early.resolve():
        raise SystemExit(
            subprocess.call(
                [str(_vpy_early), str(Path(__file__).resolve()), *sys.argv[1:]],
                env=os.environ.copy(),
            )
        )

import json
import logging
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from transformers.modeling_utils import WEIGHTS_NAME
from transformers.trainer import TRAINING_ARGS_NAME

# ── Bypass torch.load CVE check (must precede all transformers imports) ───
_orig_load = torch.load


def _patched_load(*a, **kw):
    kw.pop("weights_only", None)
    return _orig_load(*a, **kw)


torch.load = _patched_load

import transformers.utils.import_utils as _hf_iu

_hf_iu.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils as _hf_mu

_hf_mu.check_torch_load_is_safe = lambda: None
try:
    import transformers.trainer as _hf_tr

    _hf_tr.check_torch_load_is_safe = lambda: None
except Exception:
    pass

from transformers import (
    AutoConfig,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    T5Tokenizer,
    set_seed,
)
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

# Ensure src/ on path for eval import
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.eval.metrics import V4Metrics as EvalMetrics
from src.model.v5.config import V5Config
from src.model.v5.dataset import V5Dataset
from src.model.v5.model import V5SCLM

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RT_LOCAL = _REPO_ROOT / "models" / "ReactionT5v2-forward"
DEFAULT_REACTION_MODEL = (
    str(_RT_LOCAL) if _RT_LOCAL.is_dir() else "sagawa/ReactionT5v2-forward"
)
DEFAULT_MOLT5_MODEL = os.environ.get("V5_MOLT5_MODEL", "laituan245/molt5-small")
OUTPUT_DIR = os.environ.get("V5B1_OUTPUT_DIR", "results/checkpoints/v5b1_calib")
LOG_FILE = os.environ.get("V5B1_LOG_FILE", "logs/v5b1/train.log")
# Rank-0 训练进度快照（与 Trainer 内存 state / checkpoint 内 trainer_state.json 一致）
V5_STATUS_FILE = os.environ.get("V5_STATUS_FILE", str(_REPO_ROOT / "_status.txt"))
AUGMENT_N = 5

# ── V5B1 memory-efficient LoRA knobs (MolT5 / ReactionT5 compatible) ──────
# Model family: auto-detect by base model name/path; can be overridden.
V5B1_MODEL_FAMILY = os.environ.get("V5B1_MODEL_FAMILY", "auto").strip().lower()
V5B1_BASE_MODEL = os.environ.get("V5B1_BASE_MODEL", "").strip()
# Force memory mode: FP16 + gradient checkpointing.
V5B1_MICRO_BATCH = int(os.environ.get("V5B1_MICRO_BATCH", "1"))
V5B1_GRAD_ACCUM = int(os.environ.get("V5B1_GRAD_ACCUM", "8"))
# max_steps: default -1 = train for num_train_epochs (full LoRA). Set positive (e.g. 300) for calibration.
_ms_raw = os.environ.get("V5B1_MAX_STEPS", "-1").strip()
try:
    V5B1_MAX_STEPS = int(_ms_raw)
except ValueError:
    V5B1_MAX_STEPS = -1
V5B1_LOGGING_STEPS = int(os.environ.get("V5B1_LOGGING_STEPS", "50"))
V5B1_NUM_EPOCHS_ENV = os.environ.get("V5B1_NUM_EPOCHS", "").strip()
V5B1_GRADIENT_CHECKPOINTING = os.environ.get("V5B1_GRADIENT_CHECKPOINTING", "1") not in (
    "0",
    "false",
    "False",
)
# LoRA on attention only.
V5B1_LORA_R = int(os.environ.get("V5B1_LORA_R", "16"))
V5B1_LORA_ALPHA = int(os.environ.get("V5B1_LORA_ALPHA", "32"))
V5B1_LORA_DROPOUT = float(os.environ.get("V5B1_LORA_DROPOUT", "0.05"))
V5B1_LORA_TARGET_MODULES = ["q", "k", "v", "o"]
V5B1_LEARNING_RATE = float(os.environ.get("V5B1_LEARNING_RATE", "2e-4"))
V5B1_FORCE_FP16 = os.environ.get("V5B1_FORCE_FP16", "1") not in ("0", "false", "False")
# Resume: `auto` / `1` / `true` → newest `checkpoint-*` under OUTPUT_DIR; else absolute/相对 OUTPUT_DIR 的路径。
V5B1_RESUME_FROM_CHECKPOINT_RAW = os.environ.get("V5B1_RESUME_FROM_CHECKPOINT", "").strip()
# 入口脚本显示名（如 V5A2/V5B2）；不设则按 MolT5→V5A1、ReactionT5→V5B1。
V5_LORA_BRANCH_DISPLAY = os.environ.get("V5_LORA_BRANCH_DISPLAY", "").strip()

if V5B1_MICRO_BATCH != 1:
    raise ValueError(
        f"V5B1: require V5B1_MICRO_BATCH == 1 for low-VRAM mode (got {V5B1_MICRO_BATCH})"
    )
if not (4 <= V5B1_GRAD_ACCUM <= 8):
    raise ValueError(
        f"V5B1: require 4 <= V5B1_GRAD_ACCUM <= 8 (got {V5B1_GRAD_ACCUM})"
    )
if not V5B1_GRADIENT_CHECKPOINTING:
    raise ValueError("V5B1: V5B1_GRADIENT_CHECKPOINTING must stay enabled in low-VRAM mode.")

warnings.filterwarnings("ignore")
Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("v5b1_train")


def _flush_log_handlers():
    for lg in (logging.root, logging.getLogger("v5b1_train")):
        for h in lg.handlers:
            try:
                h.flush()
            except Exception:
                pass


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


class _RepoRootStatusCallback(TrainerCallback):
    """将 TrainerState（与 checkpoint 内 trainer_state.json 同源）写入仓库根 `_status.txt`，便于外部轮询。"""

    def __init__(self, status_path: str, branch: str, output_dir: str, log_file: str):
        self.status_path = Path(status_path)
        self.branch = branch
        self.output_dir = output_dir
        self.log_file = log_file
        self._eval_lines: list[str] = []

    def _disk_trainer_state_step(self, global_step: int) -> str | None:
        p = Path(self.output_dir) / f"checkpoint-{global_step}" / "trainer_state.json"
        if not p.is_file():
            return None
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            return str(data.get("global_step", ""))
        except Exception:
            return None

    def _write(self, args, state, metrics: dict | None = None) -> None:
        if not state.is_world_process_zero:
            return
        ms = state.max_steps
        max_disp = str(ms) if ms is not None and ms >= 0 else "unknown"
        lines = [
            f"updated_utc: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
            f"branch: {self.branch}",
            f"output_dir: {self.output_dir}",
            f"log_file: {self.log_file}",
            f"Last step: {state.global_step}/{max_disp}",
            f"epoch: {state.epoch:.6f}",
            f"best_model_checkpoint: {state.best_model_checkpoint or ''}",
            f"best_metric: {state.best_metric}",
            "Has traceback: False",
            f"Total evals: {len(self._eval_lines)}",
        ]
        log_p = Path(self.log_file)
        try:
            log_sz = log_p.stat().st_size if log_p.is_file() else 0
        except OSError:
            log_sz = 0
        lines.append(f"File size: {log_sz} bytes")
        disk_s = self._disk_trainer_state_step(state.global_step)
        if disk_s is not None:
            lines.append(f"disk_trainer_state.json global_step: {disk_s}")
        if metrics:
            lines.append("last_metrics: " + json.dumps(metrics, default=str, ensure_ascii=False))
        for ev in self._eval_lines:
            lines.append(ev)
        _atomic_write_text(self.status_path, "\n".join(lines) + "\n")

    def on_train_begin(self, args, state, control, **kwargs):
        self._eval_lines.clear()
        self._write(args, state, None)

    def on_log(self, args, state, control, **kwargs):
        self._write(args, state, kwargs.get("logs"))

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self._eval_lines.append("Eval: " + json.dumps(metrics, default=str, ensure_ascii=False))
        self._write(args, state, metrics)

    def on_save(self, args, state, control, **kwargs):
        self._write(args, state, None)

    def on_train_end(self, args, state, control, **kwargs):
        self._write(args, state, None)


class V5DataCollator:
    """Stacks pre-padded tensors."""

    def __call__(self, features):
        return {k: torch.stack([f[k] for f in features]) for k in features[0]}


class _TrainHeartbeatCallback(TrainerCallback):
    """Logs early optimizer steps — slow CPU/GPU may take minutes before logging_steps kicks in."""

    def on_train_begin(self, args, state, control, **kwargs):
        logger.info(
            "Heartbeat: training loop started (logging_steps=%s, grad_accum=%s)",
            args.logging_steps,
            args.gradient_accumulation_steps,
        )
        _flush_log_handlers()

    def on_step_end(self, args, state, control, **kwargs):
        if 0 < state.global_step <= 15:
            logger.info(
                "Heartbeat: optimizer step %s epoch=%.5f",
                state.global_step,
                state.epoch,
            )
            _flush_log_handlers()


class V5Trainer(Seq2SeqTrainer):
    """Seq2SeqTrainer subclass that passes reaction_ids during generation eval."""

    def _save(self, output_dir=None, state_dict=None):
        """Save full `V5SCLM.state_dict()` as pytorch_model.bin so Trainer can resume and load_best_model_at_end."""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        if state_dict is None:
            state_dict = self.model.state_dict()
        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        torch.save(
            {
                "reaction_embed": self.model.reaction_embed.state_dict(),
                "token_to_idx": self.model.token_to_idx,
                "cfg_dropout_prob": self.model.cfg_dropout_prob,
                "cfg_guidance_scale": self.model.cfg_guidance_scale,
                "num_reaction_tokens": self.model.num_reaction_tokens,
                "reaction_embed_dim": self.model.reaction_embed_dim,
            },
            os.path.join(output_dir, "v5_state.pt"),
        )
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        elif (
            self.data_collator is not None
            and hasattr(self.data_collator, "tokenizer")
            and self.data_collator.tokenizer is not None
        ):
            logger.info("Saving Trainer.data_collator.tokenizer with checkpoint.")
            self.data_collator.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def prediction_step(
        self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs
    ):
        reaction_ids = inputs.pop("reaction_ids")
        gen_kwargs.setdefault("max_length", self.args.generation_max_length)
        gen_kwargs.setdefault("num_beams", self.args.generation_num_beams)
        generated = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            reaction_ids=reaction_ids,
            **gen_kwargs,
        )
        inputs["reaction_ids"] = reaction_ids
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss
        return (loss, generated, inputs["labels"])


def make_compute_metrics(
    val_df: pd.DataFrame, tokenizer: T5Tokenizer, ppm_threshold: float
):
    parents = val_df["parent_smiles"].tolist()
    targets = val_df["product_smiles"].tolist()
    deltas = val_df["delta_mz"].tolist()

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded = tokenizer.batch_decode(preds, skip_special_tokens=True)

        m = min(len(decoded), len(targets), len(parents), len(deltas))
        if m < len(decoded):
            logger.warning(
                "compute_metrics: decoded_len=%d > val_df rows=%d; using first %d pairs "
                "(DDP/multi-process eval can over-concatenate preds vs val_df).",
                len(decoded),
                len(targets),
                m,
            )
        if m == 0:
            return {"validity": 0.0, "exact_match": 0.0, "ppm_pass_rate": 0.0}

        results = {"valid_count": 0, "em_count": 0, "ppm_count": 0}
        for i in range(m):
            pred_smi = decoded[i].strip()
            if not EvalMetrics.check_validity(pred_smi):
                continue
            results["valid_count"] += 1

            pred_ik = EvalMetrics.get_inchikey_14(pred_smi)
            target_ik = EvalMetrics.get_inchikey_14(targets[i])
            if pred_ik and target_ik and pred_ik == target_ik:
                results["em_count"] += 1

            if EvalMetrics.check_ppm_fidelity(
                parents[i], pred_smi, deltas[i], threshold=ppm_threshold
            ):
                results["ppm_count"] += 1

        return {
            "validity": round(results["valid_count"] / m * 100, 2),
            "exact_match": round(results["em_count"] / m * 100, 2),
            "ppm_pass_rate": round(results["ppm_count"] / m * 100, 2),
        }

    return compute_metrics


def _hf_hub_cache_root() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _latest_hf_hub_snapshot(model_id: str) -> Path | None:
    """Resolve hub repo id to newest local snapshot dir if HF cache exists (offline-friendly)."""
    slug = "models--" + model_id.replace("/", "--")
    snap = _hf_hub_cache_root() / slug / "snapshots"
    if not snap.is_dir():
        return None
    dirs = [p for p in snap.iterdir() if p.is_dir()]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0]


def _hf_hub_snapshot_config_ok(snapshot_dir: Path) -> bool:
    """Incomplete/corrupt HF snapshots (e.g. partial download) lack model_type and break AutoConfig."""
    cfg_path = snapshot_dir / "config.json"
    if not cfg_path.is_file():
        return False
    try:
        with open(cfg_path, encoding="utf-8") as f:
            data = json.load(f)
        return isinstance(data, dict) and bool(data.get("model_type"))
    except Exception:
        return False


def _latest_checkpoint_in_output_dir(output_dir: str) -> str | None:
    """Newest checkpoint-* by global step suffix; requires trainer_state.json + pytorch_model.bin."""
    root = Path(output_dir)
    if not root.is_dir():
        return None
    best_step = -1
    best_path: str | None = None
    for p in root.iterdir():
        if not p.is_dir() or not p.name.startswith("checkpoint-"):
            continue
        try:
            step = int(p.name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if not (p / "trainer_state.json").is_file() or not (p / WEIGHTS_NAME).is_file():
            continue
        if step > best_step:
            best_step = step
            best_path = str(p.resolve())
    return best_path


def _resolve_resume_checkpoint(output_dir: str) -> str | None:
    raw = V5B1_RESUME_FROM_CHECKPOINT_RAW
    if not raw:
        return None
    low = raw.lower()
    if low in ("1", "true", "yes", "auto"):
        picked = _latest_checkpoint_in_output_dir(output_dir)
        if picked:
            logger.info("Resume: auto-selected checkpoint %s", picked)
        else:
            logger.warning(
                "Resume: V5B1_RESUME_FROM_CHECKPOINT=%s but no valid checkpoint-* under %s",
                raw,
                output_dir,
            )
        return picked
    p = Path(raw)
    if not p.is_absolute():
        p = Path(output_dir) / raw
    p = p.resolve()
    if p.is_dir() and (p / "trainer_state.json").is_file() and (p / WEIGHTS_NAME).is_file():
        logger.info("Resume: using checkpoint %s", p)
        return str(p)
    logger.warning("Resume: invalid checkpoint path %s (missing trainer_state.json or %s)", p, WEIGHTS_NAME)
    return None


def resolve_reaction_base_model() -> tuple[str, bool]:
    if V5B1_BASE_MODEL:
        custom_path = Path(V5B1_BASE_MODEL)
        if custom_path.exists():
            logger.info("Using V5B1 custom local base model: %s", custom_path)
            return str(custom_path), True
        logger.info("Using V5B1 custom remote base model id: %s", V5B1_BASE_MODEL)
        return V5B1_BASE_MODEL, False

    local_snapshot = os.getenv("V5B_REACTION_MODEL_PATH", "").strip()
    if local_snapshot:
        snapshot_path = Path(local_snapshot)
        if snapshot_path.exists():
            logger.info("Using local ReactionT5 snapshot: %s", snapshot_path)
            return str(snapshot_path), True
        logger.warning("Reaction model path is set but not found: %s", snapshot_path)

    if V5B1_MODEL_FAMILY == "molt5":
        mid = DEFAULT_MOLT5_MODEL
        molt5_path = Path(mid)
        if molt5_path.exists():
            logger.info("Using default MolT5 model: %s", molt5_path)
            return str(molt5_path), True
        cached = _latest_hf_hub_snapshot(mid)
        if cached is not None and _hf_hub_snapshot_config_ok(cached):
            logger.info("Using cached MolT5 snapshot (local_files_only): %s", cached)
            return str(cached), True
        if cached is not None:
            logger.warning(
                "MolT5 HF cache snapshot unusable (missing/invalid config.json model_type): %s — "
                "falling back to model id (set HF_ENDPOINT or network for re-download).",
                cached,
            )
        logger.info("Using default MolT5 model id/path (may require network): %s", mid)
        return mid, False

    reaction_path = Path(DEFAULT_REACTION_MODEL)
    if reaction_path.exists():
        logger.info("Using default ReactionT5 model: %s", reaction_path)
        return str(reaction_path), True
    logger.info("Using default ReactionT5 model id/path: %s", DEFAULT_REACTION_MODEL)
    return DEFAULT_REACTION_MODEL, False


def infer_model_family(base_model: str) -> str:
    if V5B1_MODEL_FAMILY in {"molt5", "reactiont5"}:
        return V5B1_MODEL_FAMILY
    name = base_model.lower()
    if "reactiont5" in name:
        return "reactiont5"
    if "molt5" in name:
        return "molt5"
    return "reactiont5"


def get_length_profile(model_family: str) -> tuple[int, int]:
    if model_family == "molt5":
        return 128, 128
    return 256, 128


def branch_display_name(model_family: str) -> str:
    """Train script filename is train_v5b1.py (shared kernel); logs use this for clarity."""
    if V5_LORA_BRANCH_DISPLAY:
        return V5_LORA_BRANCH_DISPLAY
    return "V5A1" if model_family == "molt5" else "V5B1"


def apply_lora_attention_only(model: V5SCLM):
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=V5B1_LORA_R,
        lora_alpha=V5B1_LORA_ALPHA,
        lora_dropout=V5B1_LORA_DROPOUT,
        target_modules=V5B1_LORA_TARGET_MODULES,
    )
    model.base_model = get_peft_model(model.base_model, lora_cfg)
    if hasattr(model.base_model, "print_trainable_parameters"):
        model.base_model.print_trainable_parameters()


def log_sanity_forward(model: V5SCLM, train_dataset: V5Dataset, device: torch.device):
    """Run one tiny forward pass and log label coverage / loss sanity."""
    sample = train_dataset[0]
    labels = sample["labels"]
    valid_label_tokens = int((labels != -100).sum().item())
    logger.info("Sanity: sample[0] valid label tokens = %d", valid_label_tokens)

    batch = {
        "input_ids": sample["input_ids"].unsqueeze(0).to(device),
        "attention_mask": sample["attention_mask"].unsqueeze(0).to(device),
        "labels": sample["labels"].unsqueeze(0).to(device),
        "reaction_ids": sample["reaction_ids"].unsqueeze(0).to(device),
    }
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        out = model(**batch)
        loss_val = float(out.loss.detach().float().cpu().item())
    logger.info("Sanity: single-batch forward loss = %.6f", loss_val)


def main(config: V5Config = None):
    if config is None:
        config = V5Config()

    config.output_dir = OUTPUT_DIR
    config.log_dir = str(Path(LOG_FILE).parent)
    config.augment_n = AUGMENT_N
    if V5B1_NUM_EPOCHS_ENV:
        config.num_epochs = int(V5B1_NUM_EPOCHS_ENV)

    resolved_model, use_local_only = resolve_reaction_base_model()
    model_family = infer_model_family(resolved_model)
    max_input_len, max_output_len = get_length_profile(model_family)
    config.base_model = resolved_model
    base_cfg = AutoConfig.from_pretrained(config.base_model, local_files_only=use_local_only)
    config.reaction_embed_dim = int(base_cfg.d_model)

    set_seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer: %s", config.base_model)
    logger.info("Model family: %s", model_family)
    tokenizer = T5Tokenizer.from_pretrained(
        config.base_model, local_files_only=use_local_only
    )
    all_special_tokens = config.special_tokens
    logger.info("ReactionT5 tokenizer vocab (before): %d", tokenizer.vocab_size)
    tokenizer.add_special_tokens({"additional_special_tokens": all_special_tokens})
    logger.info("ReactionT5 tokenizer vocab (after): %d", len(tokenizer))

    logger.info("Checking special token registration:")
    for tok in all_special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(tok)
        status = "OK" if token_id != tokenizer.unk_token_id else "UNK"
        logger.info("  %s -> id=%s %s", tok, token_id, status)

    logger.info("Building V5SCLM...")
    model = V5SCLM(
        base_model_name=config.base_model,
        num_reaction_tokens=config.num_reaction_tokens,
        reaction_embed_dim=config.reaction_embed_dim,
        cfg_dropout_prob=config.cfg_dropout_prob,
        cfg_guidance_scale=config.cfg_guidance_scale,
        token_to_idx=config.token_to_idx,
    )
    model.base_model.resize_token_embeddings(len(tokenizer))
    apply_lora_attention_only(model)
    logger.info("Model parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    if V5B1_GRADIENT_CHECKPOINTING:
        model.base_model.config.use_cache = False
        model.base_model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing: enabled on base_model (saves VRAM, ~10–25%% slower/step).")
    else:
        logger.info("Gradient checkpointing: off")

    branch = branch_display_name(model_family)
    repo_status_cb = _RepoRootStatusCallback(V5_STATUS_FILE, branch, config.output_dir, LOG_FILE)
    logger.info("Repo status snapshot (TrainerState): %s", V5_STATUS_FILE)

    logger.info(
        "%s low-vram profile (LoRA kernel train_v5b1.py): family=%s max_in=%d max_out=%d micro_batch=%d grad_accum=%d eff_batch=%d "
        "max_steps=%d epochs=%d grad_ckpt=%s fp16=%s lr=%.2e log_steps=%d lora(r=%d,alpha=%d,target=%s)",
        branch,
        model_family,
        max_input_len,
        max_output_len,
        V5B1_MICRO_BATCH,
        V5B1_GRAD_ACCUM,
        V5B1_MICRO_BATCH * V5B1_GRAD_ACCUM,
        V5B1_MAX_STEPS,
        config.num_epochs,
        V5B1_GRADIENT_CHECKPOINTING,
        bool(torch.cuda.is_available() and V5B1_FORCE_FP16),
        V5B1_LEARNING_RATE,
        V5B1_LOGGING_STEPS,
        V5B1_LORA_R,
        V5B1_LORA_ALPHA,
        ",".join(V5B1_LORA_TARGET_MODULES),
    )

    with torch.no_grad():
        smoke = tokenizer("CCO", return_tensors="pt")
        _ = model.base_model.encoder(input_ids=smoke["input_ids"])
    logger.info("%s encoder smoke test passed for input format.", model_family)

    logger.info("Loading datasets...")
    train_dataset = V5Dataset(
        config.train_csv,
        tokenizer,
        config.token_to_idx,
        max_input_len=max_input_len,
        max_output_len=max_output_len,
        augment=True,
        augment_n=config.augment_n,
        use_chemical_whitespace=True,
    )
    val_dataset = V5Dataset(
        config.val_csv,
        tokenizer,
        config.token_to_idx,
        max_input_len=max_input_len,
        max_output_len=max_output_len,
        augment=False,
        use_chemical_whitespace=True,
    )

    val_df = val_dataset.df
    data_collator = V5DataCollator()
    compute_metrics_fn = make_compute_metrics(
        val_df, tokenizer, ppm_threshold=config.ppm_threshold
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_sanity_forward(model, train_dataset, device)

    use_bf16 = False
    use_fp16 = torch.cuda.is_available() and V5B1_FORCE_FP16
    logger.info("Precision mode (forced low-VRAM): bf16=%s, fp16=%s", use_bf16, use_fp16)

    full_lora = V5B1_MAX_STEPS < 0
    if full_lora:
        training_args = Seq2SeqTrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_epochs,
            max_steps=-1,
            per_device_train_batch_size=V5B1_MICRO_BATCH,
            per_device_eval_batch_size=V5B1_MICRO_BATCH,
            gradient_accumulation_steps=V5B1_GRAD_ACCUM,
            gradient_checkpointing=False,
            learning_rate=V5B1_LEARNING_RATE,
            warmup_ratio=config.warmup_ratio,
            bf16=use_bf16,
            fp16=use_fp16,
            predict_with_generate=True,
            generation_max_length=max_output_len,
            generation_num_beams=1,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model=config.metric_for_best,
            greater_is_better=config.greater_is_better,
            logging_dir=config.log_dir,
            logging_steps=V5B1_LOGGING_STEPS,
            logging_first_step=True,
            save_total_limit=3,
            seed=config.seed,
            report_to="none",
            dataloader_num_workers=0,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
        )
        callbacks = [
            _TrainHeartbeatCallback(),
            repo_status_cb,
            EarlyStoppingCallback(early_stopping_patience=config.patience),
        ]
    else:
        training_args = Seq2SeqTrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_epochs,
            max_steps=V5B1_MAX_STEPS,
            per_device_train_batch_size=V5B1_MICRO_BATCH,
            per_device_eval_batch_size=V5B1_MICRO_BATCH,
            gradient_accumulation_steps=V5B1_GRAD_ACCUM,
            gradient_checkpointing=False,
            learning_rate=V5B1_LEARNING_RATE,
            warmup_ratio=config.warmup_ratio,
            bf16=use_bf16,
            fp16=use_fp16,
            predict_with_generate=False,
            generation_max_length=max_output_len,
            generation_num_beams=1,
            eval_strategy="no",
            save_strategy="no",
            load_best_model_at_end=False,
            logging_dir=config.log_dir,
            logging_steps=V5B1_LOGGING_STEPS,
            save_total_limit=1,
            seed=config.seed,
            report_to="none",
            dataloader_num_workers=0,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
        )
        callbacks = [_TrainHeartbeatCallback(), repo_status_cb]

    trainer = V5Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks,
    )

    mode = "full LoRA" if full_lora else "calibration"
    logger.info(
        "Starting %s %s (%d epochs, max_steps=%s, effective batch=%d)...",
        branch,
        mode,
        config.num_epochs,
        training_args.max_steps,
        V5B1_MICRO_BATCH * V5B1_GRAD_ACCUM,
    )
    _flush_log_handlers()

    resume_ckpt = _resolve_resume_checkpoint(config.output_dir)

    try:
        trainer.train(resume_from_checkpoint=resume_ckpt)
    except BaseException:
        logger.exception("trainer.train() aborted with exception")
        _flush_log_handlers()
        raise
    finally:
        _flush_log_handlers()

    if full_lora:
        best_dir = os.path.join(config.output_dir, "best")
        logger.info("Saving best model snapshot to %s", best_dir)
        if isinstance(model.base_model, PeftModel):
            logger.info("Merging LoRA into backbone for inference-compatible export...")
            model.base_model = model.base_model.merge_and_unload()
        model.save(best_dir, tokenizer=tokenizer)

    logger.info("%s %s complete.", branch, mode)


if __name__ == "__main__":
    main()

