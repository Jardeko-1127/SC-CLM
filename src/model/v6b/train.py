"""
V6b training script — generates OpenNMT YAML config and launches onmt_train.

Usage:
  python src/model/v6b/build_vocab.py   # first: build vocabularies
  python src/model/v6b/train.py         # then: generate config + train
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import yaml

from src.model.v6b.config import V6bConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("v6b_train")

REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_venv_python() -> str:
    venv_python = REPO_ROOT / "venv" / "Scripts" / "python.exe"
    if venv_python.is_file():
        return str(venv_python)
    return sys.executable


def generate_yaml(config: V6bConfig) -> Path:
    """Generate OpenNMT YAML config file."""
    data_dir = REPO_ROOT / config.data_dir
    model_dir = REPO_ROOT / config.model_dir
    log_dir = REPO_ROOT / config.log_dir

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    yaml_config = {
        # Data
        "save_data": str(data_dir / "opennmt"),
        "src_vocab": str(data_dir / "src_vocab.txt"),
        "tgt_vocab": str(data_dir / "tgt_vocab.txt"),
        "data": {
            "train": {
                "path_src": str(data_dir / "src-train.txt"),
                "path_tgt": str(data_dir / "tgt-train.txt"),
            },
            "valid": {
                "path_src": str(data_dir / "src-val.txt"),
                "path_tgt": str(data_dir / "tgt-val.txt"),
            },
        },
        # Model
        "save_model": str(model_dir / "model"),
        "encoder_type": "transformer",
        "decoder_type": "transformer",
        "enc_layers": config.num_layers,
        "dec_layers": config.num_layers,
        "heads": config.num_heads,
        "hidden_size": config.hidden_size,
        "word_vec_size": config.word_vec_size,
        "transformer_ff": config.ffn_size,
        "dropout_steps": [0],
        "dropout": [config.dropout],
        "attention_dropout": [config.dropout],
        # Training
        "batch_size": config.batch_size,
        "batch_type": config.batch_type,
        "normalization": "tokens",
        "accum_count": [config.accum_count],
        "optim": "adam",
        "learning_rate": config.learning_rate,
        "warmup_steps": config.warmup_steps,
        "decay_method": "noam",
        "max_grad_norm": 5.0,
        "label_smoothing": 0.1,
        "param_init": 0.0,
        "param_init_glorot": True,
        # Training steps
        "train_steps": config.train_steps,
        "valid_steps": config.valid_steps,
        "save_checkpoint_steps": config.save_checkpoint_steps,
        "keep_checkpoint": config.keep_checkpoint,
        "seed": config.seed,
        "report_every": 100,
        # Logging
        "tensorboard": True,
        "tensorboard_log_dir": str(log_dir),
        # GPU
        "world_size": 1,
        "gpu_ranks": [0],
    }

    yaml_path = model_dir / "v6b_config.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_config, f, default_flow_style=False, allow_unicode=True)

    logger.info("YAML config written to %s", yaml_path)
    return yaml_path


def build_opennmt_vocab(config: V6bConfig):
    """Build OpenNMT binary vocabulary using onmt_build_vocab."""
    data_dir = REPO_ROOT / config.data_dir
    yaml_path = REPO_ROOT / config.model_dir / "v6b_config.yaml"

    if not yaml_path.exists():
        generate_yaml(config)

    logger.info("Building OpenNMT binary vocab...")
    cmd = [
        _resolve_venv_python(), "-m", "onmt.bin.build_vocab",
        "-config", str(yaml_path),
    ]
    logger.info("  %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
        logger.info("Vocabulary built successfully.")
    except subprocess.CalledProcessError as e:
        logger.error("Build vocab failed (exit %d). Check data format.", e.returncode)
        raise


def train():
    """Main: generate config, build vocab, start training."""
    config = V6bConfig()

    # Generate YAML
    yaml_path = generate_yaml(config)

    # Build OpenNMT vocab (if not exists)
    save_data = REPO_ROOT / config.data_dir / "opennmt.vocab.pt"
    if not save_data.exists():
        build_opennmt_vocab(config)
    else:
        logger.info("OpenNMT vocab already exists: %s", save_data)

    # Launch training
    logger.info("Starting OpenNMT training...")
    cmd = [
        _resolve_venv_python(), "-m", "onmt.bin.train",
        "-config", str(yaml_path),
    ]
    logger.info("  %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    except subprocess.CalledProcessError as e:
        logger.error("Training failed (exit %d).", e.returncode)
        raise


if __name__ == "__main__":
    train()
