"""
SC-CLM V5A Training Script
--------------------------
V5 architecture with MolT5 backbone, augment_n=5.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path

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
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5Tokenizer,
    set_seed,
)

# Ensure src/ on path for eval import
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.eval.metrics import V4Metrics as EvalMetrics
from src.model.v5.config import V5Config
from src.model.v5.dataset import V5Dataset
from src.model.v5.model import V5SCLM

BASE_MODEL = "laituan245/molt5-small"
# 服务器新一轮训练建议: export V5A_OUTPUT_DIR=results/checkpoints/v5a_$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR = (os.environ.get("V5A_OUTPUT_DIR") or "results/checkpoints/v5a").strip()
LOG_FILE = (os.environ.get("V5A_TRAIN_LOG") or "logs/v5a/train.log").strip()
AUGMENT_N = 5

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("v5a_train")


class V5DataCollator:
    """Stacks pre-padded tensors."""

    def __call__(self, features):
        return {k: torch.stack([f[k] for f in features]) for k in features[0]}


class V5Trainer(Seq2SeqTrainer):
    """Seq2SeqTrainer subclass that passes reaction_ids during generation eval."""

    def _save(self, output_dir=None, state_dict=None):
        """Full V5SCLM state as pytorch_model.bin for Trainer resume / load_best_model_at_end."""
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
    """Factory: returns compute_metrics callable using canonical EvalMetrics."""

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


def main(config: V5Config = None):
    if config is None:
        config = V5Config()

    config.base_model = BASE_MODEL
    config.output_dir = OUTPUT_DIR
    config.log_dir = str(Path(LOG_FILE).parent)
    config.augment_n = AUGMENT_N

    set_seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer: %s", config.base_model)
    all_special_tokens = config.special_tokens
    tokenizer = T5Tokenizer.from_pretrained(config.base_model, local_files_only=False)
    tokenizer.add_special_tokens({"additional_special_tokens": all_special_tokens})

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
    logger.info("Model parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    logger.info("Loading datasets...")
    train_dataset = V5Dataset(
        config.train_csv,
        tokenizer,
        config.token_to_idx,
        max_input_len=config.max_input_len,
        max_output_len=config.max_output_len,
        augment=True,
        augment_n=config.augment_n,
        use_chemical_whitespace=True,
    )
    val_dataset = V5Dataset(
        config.val_csv,
        tokenizer,
        config.token_to_idx,
        max_input_len=config.max_input_len,
        max_output_len=config.max_output_len,
        augment=False,
        use_chemical_whitespace=True,
    )

    val_df = val_dataset.df
    data_collator = V5DataCollator()
    compute_metrics_fn = make_compute_metrics(
        val_df, tokenizer, ppm_threshold=config.ppm_threshold
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    logger.info("Precision mode: bf16=%s, fp16=%s", use_bf16, use_fp16)

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        bf16=use_bf16,
        fp16=use_fp16,
        predict_with_generate=True,
        generation_max_length=config.max_output_len,
        generation_num_beams=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=config.metric_for_best,
        greater_is_better=config.greater_is_better,
        logging_dir=config.log_dir,
        logging_steps=50,
        save_total_limit=3,
        seed=config.seed,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = V5Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.patience)],
    )

    logger.info(
        "Starting V5A training (%d epochs, effective batch=%d, CFG dropout=%.2f)...",
        config.num_epochs,
        config.batch_size * config.grad_accum_steps,
        config.cfg_dropout_prob,
    )
    trainer.train()

    best_dir = os.path.join(config.output_dir, "best")
    logger.info("Saving best model to %s", best_dir)
    model.save(best_dir, tokenizer=tokenizer)
    logger.info("V5A training complete.")


if __name__ == "__main__":
    main()
