# SC-CLM

Seed-Conditioned Chemical Language Model for environmental transformation product prediction.

## Overview

SC-CLM predicts `parent SMILES -> product SMILES` under reaction constraints.  
Current development focuses on V5 with embedding-conditioned encoder-decoder architecture and CFG.

## Active variants

- **V5A**: MolT5-Small backbone (`src/model/v5/train_v5a.py`)
- **V5B**: ReactionT5v2-forward backbone (`src/model/v5/train_v5b.py`)
- **V5A1**: MolT5-Small + LoRA calibration (`src/model/v5/train_v5a1.py`)
- **V5B1**: ReactionT5v2-forward + LoRA calibration (`src/model/v5/train_v5b1.py`)

## Core stack

- Python 3.12 (`venv`)
- PyTorch + HuggingFace Transformers
- RDKit (chemical validity and descriptors)
- Pandas / NumPy

## Quick start

```bash
# preprocess
python src/data/preprocess.py

# train
python src/model/v5/train_v5a.py
python src/model/v5/train_v5b.py
python src/model/v5/train_v5a1.py
python src/model/v5/train_v5b1.py

# inference / eval
python src/model/v5/inference.py
python src/eval/metrics.py
```

## Important paths

- Main guidance: `CLAUDE.md`
- V5A logs: `logs/v5a/train.log`
- V5A checkpoints: `results/checkpoints/v5a/`
- V5A1 logs/checkpoints: `logs/v5a1/train.log`, `results/checkpoints/v5a1_calib/`
- V5B1 logs/checkpoints: `logs/v5b1/train.log`, `results/checkpoints/v5b1_calib/`
- Reference archives: `ref/`
