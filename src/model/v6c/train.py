"""V6c training — minimal diff from V6a. Only changes: config/class names + output paths."""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoConfig, Seq2SeqTrainingArguments, T5Tokenizer, EarlyStoppingCallback,
)
from transformers.utils import WEIGHTS_NAME
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from src.eval.metrics import V4Metrics as EvalMetrics
from src.model.v6.train import (                    # ← reuse V6a helpers
    V6Trainer, make_compute_metrics, _resolve_base_model_path, set_seed,
)
from src.model.v6.dataset import V6Dataset
from src.model.v6c.config import V6cConfig           # ← V6c
from src.model.v6c.model import V6cSCLM              # ← V6c

TRAINING_ARGS_NAME = "training_args.bin"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("v6c_train")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ── V6cTrainer: override _save to write v6c_state.pt ──────────────────────
class V6cTrainer(V6Trainer):
    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if state_dict is None:
            state_dict = self.model.state_dict()
        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))

        m = self.model
        state = {
            "cfg_dropout_prob": m.cfg_dropout_prob,
            "cfg_guidance_scale": m.cfg_guidance_scale,
            "mass_embed_dim": m.mass_embed_dim,
            "mass_mode": m.mass_mode,
            "mass_mlp_hidden": m.v6_config.mass_mlp_hidden,
            "mass_bins": m.v6_config.mass_bins,
        }
        if m.mass_mode == "mlp":
            state["mass_mlp"] = m.mass_mlp.state_dict()
        else:
            state["mass_embed"] = m.mass_embed.state_dict()
            state["num_mass_bins"] = m.v6_config.num_mass_bins
        torch.save(state, os.path.join(output_dir, "v6c_state.pt"))

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


def apply_lora(model: V6cSCLM):
    c = model.v6_config
    lc = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False,
                     r=c.lora_r, lora_alpha=c.lora_alpha, lora_dropout=c.lora_dropout,
                     target_modules=c.lora_target_modules)
    model.base_model = get_peft_model(model.base_model, lc)
    if hasattr(model.base_model, "print_trainable_parameters"):
        model.base_model.print_trainable_parameters()


# ── Main (same structure as V6a, only class names + paths changed) ────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mass_mode", default="discrete", choices=["discrete", "mlp"])
    args = parser.parse_args()

    config = V6cConfig(mass_mode=args.mass_mode)       # ← V6c
    config.base_model = _resolve_base_model_path()
    config.output_dir = "results/checkpoints/v6c"       # ← v6c path
    config.log_dir = "logs/v6c"                         # ← v6c path

    if os.environ.get("SC_CLM_SERVER", "").strip() == "1":
        config.micro_batch = int(os.environ.get("V6_MICRO_BATCH", "4"))
        config.grad_accum = int(os.environ.get("V6_GRAD_ACCUM", "4"))
        config.num_epochs = int(os.environ.get("V6_NUM_EPOCHS", "20"))
        config.patience = int(os.environ.get("V6_PATIENCE", "5"))
        config._gradient_checkpointing = False
        config._dataloader_workers = int(os.environ.get("V6_DATALOADER_WORKERS", "4"))
        logger.info("[SERVER] batch=%d accum=%d epochs=%d mass_mode=%s",
                     config.micro_batch, config.grad_accum, config.num_epochs, config.mass_mode)
    else:
        config._gradient_checkpointing = True
        config._dataloader_workers = 0

    set_seed(config.seed)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    lf = Path(config.log_dir) / f"train_{datetime.now():%Y%m%d_%H%M%S}_{config.mass_mode}.log"
    fh = logging.FileHandler(str(lf), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)
    logger.info("Log: %s  mass_mode=%s", lf, config.mass_mode)

    tokenizer = T5Tokenizer.from_pretrained(config.base_model)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[MASS]"]})
    config.mass_embed_dim = int(AutoConfig.from_pretrained(config.base_model).d_model)
    logger.info("d_model=%d", config.mass_embed_dim)

    model = V6cSCLM(config)                             # ← V6c
    model.base_model.resize_token_embeddings(len(tokenizer))
    apply_lora(model)

    if config._gradient_checkpointing:
        model.base_model.config.use_cache = False
        model.base_model.gradient_checkpointing_enable()

    train_ds = V6Dataset(config.train_csv, tokenizer, config,
                         max_input_len=config.max_input_len,
                         max_output_len=config.max_output_len,
                         augment=True, augment_n=5)
    val_ds = V6Dataset(config.val_csv, tokenizer, config,
                       max_input_len=config.max_input_len,
                       max_output_len=config.max_output_len, augment=False)
    val_df = pd.read_csv(config.val_csv)

    ta = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.micro_batch,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.learning_rate, warmup_ratio=config.warmup_ratio,
        num_train_epochs=config.num_epochs,
        logging_dir=config.log_dir, logging_strategy="steps", logging_steps=50,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model=config.metric_for_best,
        greater_is_better=config.greater_is_better,
        predict_with_generate=True,
        generation_max_length=config.max_output_len, generation_num_beams=1,
        fp16=True, dataloader_num_workers=config._dataloader_workers,
        seed=config.seed, report_to=["none"],
    )

    trainer = V6cTrainer(                              # ← V6cTrainer
        model=model, args=ta,
        train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=make_compute_metrics(val_df, tokenizer, config.ppm_threshold),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.patience)],
    )

    logger.info("V6c training start (mass_mode=%s, %d epochs)", config.mass_mode, config.num_epochs)
    trainer.train()

    best = Path(config.output_dir) / "best"
    if isinstance(model.base_model, PeftModel):
        model.base_model = model.base_model.merge_and_unload()
    model.save(str(best), tokenizer=tokenizer)
    logger.info("V6c best → %s", best)


if __name__ == "__main__":
    main()
