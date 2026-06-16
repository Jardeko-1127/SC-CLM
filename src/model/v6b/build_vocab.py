"""
V6b vocabulary builder — char-level vocab from space-separated SMILES training data.

Generates:
  data/processed/v6b/src_vocab.txt
  data/processed/v6b/tgt_vocab.txt
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

from src.model.v6b.config import V6bConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("v6b_build_vocab")

REPO_ROOT = Path(__file__).resolve().parents[3]

SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<blank>", "|"]


def _collect_tokens(filepath: Path) -> Counter:
    c = Counter()
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            c.update(tokens)
    return c


def build_vocab():
    config = V6bConfig()
    data_dir = REPO_ROOT / config.data_dir

    src_train = data_dir / "src-train.txt"
    tgt_train = data_dir / "tgt-train.txt"

    if not src_train.exists() or not tgt_train.exists():
        logger.error("Training data not found. Run prepare_data.py first.")
        return

    logger.info("Collecting source tokens...")
    src_counter = _collect_tokens(src_train)
    logger.info("  %d unique tokens", len(src_counter))

    logger.info("Collecting target tokens...")
    tgt_counter = _collect_tokens(tgt_train)
    logger.info("  %d unique tokens", len(tgt_counter))

    # Build vocab lists (special tokens first, then by frequency)
    def _build_vocab_lines(counter: Counter) -> list[str]:
        tokens = [t for t in SPECIAL_TOKENS]
        for tok, _ in counter.most_common():
            if tok not in SPECIAL_TOKENS:
                tokens.append(tok)
        return tokens

    src_vocab = _build_vocab_lines(src_counter)
    tgt_vocab = _build_vocab_lines(tgt_counter)

    src_vocab_path = data_dir / "src_vocab.txt"
    tgt_vocab_path = data_dir / "tgt_vocab.txt"

    src_vocab_path.write_text("\n".join(src_vocab) + "\n", encoding="utf-8")
    tgt_vocab_path.write_text("\n".join(tgt_vocab) + "\n", encoding="utf-8")

    logger.info("Source vocab: %d tokens → %s", len(src_vocab), src_vocab_path)
    logger.info("Target vocab: %d tokens → %s", len(tgt_vocab), tgt_vocab_path)


if __name__ == "__main__":
    build_vocab()
