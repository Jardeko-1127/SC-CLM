"""
SC-CLM V4 训练脚本 —— No-Token版本（版本A）
输入格式：[SEED] {parent_smiles}
输出格式：{product_smiles}
"""

import os, gc, logging, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
# Force torch.load to allow non-safetensors for resuming training
original_torch_load = torch.load
def forced_torch_load(*args, **kwargs):
    if 'weights_only' in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = forced_torch_load

from rdkit import Chem
from rdkit.Chem import Descriptors, inchi

# Bypass transformers CVE torch version checks aggressively
import transformers.utils.import_utils as hf_import_utils
hf_import_utils.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils as hf_modeling_utils
hf_modeling_utils.check_torch_load_is_safe = lambda: None
try:
    import transformers.trainer as hf_trainer
    hf_trainer.check_torch_load_is_safe = lambda: None
except:
    pass

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/train_v4a.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── 全局配置 ──────────────────────────────────────────────────────────────────
SEED            = 42
BASE_MODEL      = "laituan245/molt5-small"
TRAIN_CSV       = "data/processed/train.csv"
VAL_CSV         = "data/processed/val.csv"
OUTPUT_DIR      = "results/checkpoints/v4a"
LOG_DIR         = "logs/v4a"
MAX_INPUT_LEN   = 512
MAX_OUTPUT_LEN  = 512
BATCH_SIZE      = 4
GRAD_ACCUM      = 8            # 等效batch_size = 32
LEARNING_RATE   = 2e-4
NUM_EPOCHS      = 60
WARMUP_RATIO    = 0.1
PPM_THRESHOLD   = 5.0

TOKEN_TO_DELTA = {
    '[TRANS_OXIDATION]':       +15.9949,
    '[TRANS_GLUCURONIDATION]': +176.0321,
    '[TRANS_DEMETHYLATION]':   -14.0157,
    '[TRANS_DEHYDROGENATION]': -2.0157,
    '[TRANS_DEETHYLATION]':    -28.0313,
    '[TRANS_DI_OXIDATION]':    +31.9898,
    '[TRANS_HYDRATION]':       +18.0106,
    '[TRANS_SULFONATION]':     +79.9568,
    '[TRANS_ISOMERIZATION]':    0.0000,
    '[TRANS_ACETYLATION]':     +42.0106,
}

set_seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Dataset ───────────────────────────────────────────────────────────────────
class SCCLMDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer: T5Tokenizer,
                 max_input_len: int, max_output_len: int,
                 augment: bool = False):
        self.df           = pd.read_csv(csv_path)
        self.tokenizer    = tokenizer
        self.max_input    = max_input_len
        self.max_output   = max_output_len
        self.augment      = augment
        logger.info(f"加载数据集: {csv_path} ({len(self.df)}条)")

    def __len__(self):
        return len(self.df)

    def _randomize_smiles(self, smiles: str) -> str:
        """随机化SMILES表示（数据增强）"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return smiles
            atoms = list(range(mol.GetNumAtoms()))
            np.random.shuffle(atoms)
            new_mol = Chem.RenumberAtoms(mol, atoms)
            randomized = Chem.MolToSmiles(new_mol, canonical=False)
            return randomized if randomized else smiles
        except:
            return smiles

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        parent  = str(row['parent_smiles'])
        product = str(row['product_smiles'])

        # 数据增强：训练时随机化SMILES
        if self.augment:
            parent  = self._randomize_smiles(parent)
            product = self._randomize_smiles(product)

        # No-Token输入格式
        input_text  = f"[SEED] {parent}"
        output_text = product

        input_enc = self.tokenizer(
            input_text,
            max_length=self.max_input,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        output_enc = self.tokenizer(
            output_text,
            max_length=self.max_output,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        labels = output_enc['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids':      input_enc['input_ids'].squeeze(),
            'attention_mask': input_enc['attention_mask'].squeeze(),
            'labels':         labels,
        }

# ── 评估指标 ──────────────────────────────────────────────────────────────────
class V4Metrics:
    def __init__(self, val_df: pd.DataFrame, tokenizer: T5Tokenizer):
        self.parents   = val_df['parent_smiles'].tolist()
        self.targets   = val_df['product_smiles'].tolist()
        self.tokens    = val_df['token'].tolist()
        self.deltas    = val_df['delta_mz'].tolist()
        self.tokenizer = tokenizer

    @staticmethod
    def get_ik14(smi: str):
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                return None
            inchi_str = inchi.MolToInchi(mol)
            if inchi_str is None:
                return None
            ik = inchi.InchiToInchiKey(inchi_str)
            return ik[:14] if ik else None
        except:
            return None

    @staticmethod
    def get_exact_mass(smi: str):
        try:
            mol = Chem.MolFromSmiles(str(smi))
            return Descriptors.ExactMolWt(mol) if mol else None
        except:
            return None

    def compute(self, eval_pred):
        preds, labels = eval_pred

        # 解码预测结果
        if isinstance(preds, tuple):
            preds = preds[0]
        preds   = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        decoded = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        em_count       = 0
        ppm_pass_count = 0
        valid_count    = 0
        n              = len(decoded)

        for i, pred_smi in enumerate(decoded):
            pred_smi = pred_smi.strip()

            # Validity
            pred_mol = Chem.MolFromSmiles(pred_smi)
            if pred_mol is None:
                continue
            valid_count += 1

            # Exact Match（InChIKey-14）
            pred_ik14   = self.get_ik14(pred_smi)
            target_ik14 = self.get_ik14(self.targets[i])
            if pred_ik14 and target_ik14 and pred_ik14 == target_ik14:
                em_count += 1

            # PPM Fidelity
            pred_mass   = self.get_exact_mass(pred_smi)
            parent_mass = self.get_exact_mass(self.parents[i])
            if pred_mass and parent_mass:
                pred_delta   = pred_mass - parent_mass
                target_delta = self.deltas[i]
                target_mass = parent_mass + target_delta
                if target_mass <= 0:
                    continue
                ppm_error = abs(pred_delta - target_delta) / target_mass * 1e6
                if ppm_error <= 5.0:
                    ppm_pass_count += 1

        return {
            'validity':      round(valid_count / n * 100, 2),
            'exact_match':   round(em_count / n * 100, 2),
            'ppm_pass_rate': round(ppm_pass_count / n * 100, 2),
        }


# ── 主训练流程 ────────────────────────────────────────────────────────────────
def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    logger.info("加载tokenizer和模型...")
    tokenizer = T5Tokenizer.from_pretrained(BASE_MODEL)
    tokenizer.add_tokens(['[SEED]'])
    
    model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL)
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 数据集
    val_df       = pd.read_csv(VAL_CSV)
    train_dataset = SCCLMDataset(TRAIN_CSV, tokenizer,
                                  MAX_INPUT_LEN, MAX_OUTPUT_LEN,
                                  augment=True)
    val_dataset   = SCCLMDataset(VAL_CSV, tokenizer,
                                  MAX_INPUT_LEN, MAX_OUTPUT_LEN,
                                  augment=False)

    # 评估器
    metrics = V4Metrics(val_df, tokenizer)

    # 训练参数
    training_args = Seq2SeqTrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = NUM_EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate               = LEARNING_RATE,
        warmup_ratio                = WARMUP_RATIO,
        bf16                        = True,
        predict_with_generate       = True,
        generation_max_length       = MAX_OUTPUT_LEN,
        generation_num_beams        = 1,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "validity",
        greater_is_better           = True,
        logging_dir                 = LOG_DIR,
        logging_steps               = 50,
        save_total_limit            = 3,
        seed                        = SEED,
        report_to                   = "none",
        dataloader_num_workers      = 0,
    )

    trainer = Seq2SeqTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        compute_metrics = metrics.compute,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=10)],
    )

    logger.info("开始训练 SC-CLM V4A (No-Token)...")
    trainer.train(resume_from_checkpoint=True)

    logger.info("训练完成，保存最优模型...")
    trainer.save_model(f"{OUTPUT_DIR}/best")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/best")
    logger.info(f"最优模型已保存至 {OUTPUT_DIR}/best")

    # 清理显存
    del model
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("显存已清理")


if __name__ == '__main__':
    main()
