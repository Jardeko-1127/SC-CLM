"""
V6a LoRA training script (V5B1 style) — low-VRAM mass-conditioned CLM.

Usage:
  python src/model/v6/train.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from transformers import (
    AutoConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5Tokenizer,
    EarlyStoppingCallback,
)
from transformers.utils import WEIGHTS_NAME
TRAINING_ARGS_NAME = "training_args.bin"  # removed from transformers public API in 5.x

from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from src.eval.metrics import V4Metrics as EvalMetrics
from src.model.v6.config import V6Config
from src.model.v6.model import V6SCLM
from src.model.v6.dataset import V6Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("v6a_train")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ── V6Trainer ──────────────────────────────────────────────────────────────
class V6Trainer(Seq2SeqTrainer):
    """Seq2SeqTrainer subclass that passes delta_mz during generation eval."""

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        if state_dict is None:
            state_dict = self.model.state_dict()
        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        torch.save(
            {
                "mass_embed": self.model.mass_embed.state_dict(),
                "cfg_dropout_prob": self.model.cfg_dropout_prob,
                "cfg_guidance_scale": self.model.cfg_guidance_scale,
                "num_mass_bins": self.model.v6_config.num_mass_bins,
                "mass_embed_dim": self.model.mass_embed_dim,
                "mass_bins": self.model.v6_config.mass_bins,
            },
            os.path.join(output_dir, "v6_state.pt"),
        )
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def prediction_step(
        self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs
    ):
        delta_mz = inputs.pop("delta_mz")
        gen_kwargs.setdefault("max_length", self.args.generation_max_length)
        gen_kwargs.setdefault("num_beams", self.args.generation_num_beams)
        generated = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            delta_mz=delta_mz,
            **gen_kwargs,
        )
        inputs["delta_mz"] = delta_mz
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss
        return (loss, generated, inputs["labels"])


def make_compute_metrics(val_df: pd.DataFrame, tokenizer: T5Tokenizer, ppm_threshold: float):
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


# ── LoRA helpers ───────────────────────────────────────────────────────────
def _resolve_base_model_path() -> str:
    """Use local ReactionT5 if available, else HuggingFace."""
    local = Path(__file__).resolve().parents[3] / "models" / "ReactionT5v2-forward"
    if local.is_dir():
        logger.info("Using local model: %s", local)
        return str(local)
    logger.info("Using HuggingFace: sagawa/ReactionT5v2-forward")
    return "sagawa/ReactionT5v2-forward"


def apply_lora_attention_only(model: V6SCLM):
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=model.v6_config.lora_r,
        lora_alpha=model.v6_config.lora_alpha,
        lora_dropout=model.v6_config.lora_dropout,
        target_modules=model.v6_config.lora_target_modules,
    )
    model.base_model = get_peft_model(model.base_model, lora_cfg)
    if hasattr(model.base_model, "print_trainable_parameters"):
        model.base_model.print_trainable_parameters()


# ── Main ───────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    config = V6Config()
    config.base_model = _resolve_base_model_path()

    # Server-mode overrides for high-VRAM GPUs (RTX 3090 / A100)
    if os.environ.get("SC_CLM_SERVER", "").strip() == "1":
        config.micro_batch = int(os.environ.get("V6_MICRO_BATCH", "4"))
        config.grad_accum = int(os.environ.get("V6_GRAD_ACCUM", "4"))
        config.num_epochs = int(os.environ.get("V6_NUM_EPOCHS", "20"))
        config.patience = int(os.environ.get("V6_PATIENCE", "5"))
        config._gradient_checkpointing = False
        config._dataloader_workers = int(os.environ.get("V6_DATALOADER_WORKERS", "4"))
        logger.info("[SERVER MODE] batch=%d accum=%d epochs=%d workers=%d",
                    config.micro_batch, config.grad_accum, config.num_epochs, config._dataloader_workers)
    else:
        config._gradient_checkpointing = True
        config._dataloader_workers = 0

    set_seed(config.seed)

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    # File log handler (in addition to stdout)
    from datetime import datetime
    log_file = Path(config.log_dir) / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)
    logger.info("Log file: %s", log_file)

    # Load tokenizer
    logger.info("Loading tokenizer: %s", config.base_model)
    tokenizer = T5Tokenizer.from_pretrained(config.base_model)
    vocab_before = tokenizer.vocab_size
    tokenizer.add_special_tokens({"additional_special_tokens": ["[MASS]"]})
    logger.info("Tokenizer vocab: %d → %d", vocab_before, len(tokenizer))

    # Resolve d_model from backbone config
    base_cfg = AutoConfig.from_pretrained(config.base_model)
    config.mass_embed_dim = int(base_cfg.d_model)
    logger.info("Backbone d_model = %d", config.mass_embed_dim)

    # Build model
    logger.info("Building V6SCLM (%d mass bins)...", config.num_mass_bins)
    model = V6SCLM(config)
    model.base_model.resize_token_embeddings(len(tokenizer))
    apply_lora_attention_only(model)
    logger.info("Model parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    # Gradient checkpointing for VRAM
    if config._gradient_checkpointing:
        model.base_model.config.use_cache = False
        model.base_model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing: enabled")
    else:
        model.base_model.config.use_cache = True
        logger.info("Gradient checkpointing: disabled (server mode)")

    # Datasets
    train_dataset = V6Dataset(
        config.train_csv,
        tokenizer,
        config,
        max_input_len=config.max_input_len,
        max_output_len=config.max_output_len,
        augment=True,
        augment_n=5,
    )
    val_dataset = V6Dataset(
        config.val_csv,
        tokenizer,
        config,
        max_input_len=config.max_input_len,
        max_output_len=config.max_output_len,
        augment=False,
    )

    val_df = pd.read_csv(config.val_csv)

    # Training arguments
    effective_batch = config.micro_batch * config.grad_accum
    logger.info("Effective batch size = %d", effective_batch)

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.micro_batch,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        num_train_epochs=config.num_epochs,
        TENSORBOARD_LOGGING_DIR=config.log_dir,
        logging_strategy="steps",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model=config.metric_for_best,
        greater_is_better=config.greater_is_better,
        predict_with_generate=True,
        generation_max_length=config.max_output_len,
        generation_num_beams=1,
        fp16=True,
        dataloader_num_workers=config._dataloader_workers,
        seed=config.seed,
        report_to=["none"],
    )

    trainer = V6Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,  # transformers 5.x renamed tokenizer→processing_class
        compute_metrics=make_compute_metrics(val_df, tokenizer, config.ppm_threshold),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.patience)],
    )

    logger.info("Starting training (%d epochs)...", config.num_epochs)
    trainer.train()

    # Save best model (merge LoRA first)
    best_dir = Path(config.output_dir) / "best"
    if isinstance(model.base_model, PeftModel):
        logger.info("Merging LoRA into backbone for inference export...")
        model.base_model = model.base_model.merge_and_unload()
    model.save(str(best_dir), tokenizer=tokenizer)
    logger.info("Best model saved to %s", best_dir)
    logger.info("V6a training complete.")


if __name__ == "__main__":
    main()
