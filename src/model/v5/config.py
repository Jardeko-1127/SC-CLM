"""V5.0 configuration dataclass — single source of truth for all hyperparameters."""

from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict


@dataclass
class V5Config:
    """Centralized configuration for SC-CLM V5.0 training and inference."""

    # ── Model ──────────────────────────────────────────────────────────────
    base_model: str = "laituan245/molt5-small"
    num_reaction_tokens: int = 18
    reaction_embed_dim: int = 512          # must match T5 d_model

    # ── CFG (Classifier-Free Guidance) ─────────────────────────────────────
    cfg_dropout_prob: float = 0.15         # training dropout rate
    cfg_guidance_scale: float = 1.5        # inference guidance strength

    # ── Training ───────────────────────────────────────────────────────────
    batch_size: int = 4
    grad_accum_steps: int = 8              # effective batch = 32
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    num_epochs: int = 60
    max_input_len: int = 512
    max_output_len: int = 512
    seed: int = 42

    # ── Augmentation ──────────────────────────────────────────────────────
    augment_n: int = 5                      # SMILES randomization multiplier

    # ── Early stopping ────────────────────────────────────────────────────
    patience: int = 15
    metric_for_best: str = "ppm_pass_rate"
    greater_is_better: bool = True

    # ── Paths ─────────────────────────────────────────────────────────────
    train_csv: str = "data/processed/train.csv"
    val_csv: str = "data/processed/val.csv"
    output_dir: str = "results/checkpoints/v5"
    log_dir: str = "logs/v5"

    # ── Reaction token mapping (Δm/z, Da) ─────────────────────────────────
    token_to_delta: Dict[str, float] = field(default_factory=lambda: {
        "[TRANS_OXIDATION]":        15.9949,
        "[TRANS_DI_OXIDATION]":     31.9898,
        "[TRANS_DEHYDROGENATION]":  -2.0157,
        "[TRANS_REDUCTION]":         2.0157,
        "[TRANS_DEOXYGENATION]":   -15.9949,
        "[TRANS_NITRO_REDUCTION]": -30.0106,
        "[TRANS_DEMETHYLATION]":   -14.0157,
        "[TRANS_DEETHYLATION]":    -28.0313,
        "[TRANS_DEACETYLATION]":   -42.0106,
        "[TRANS_METHYLATION]":      14.0157,
        "[TRANS_HYDRATION]":        18.0106,
        "[TRANS_DEHYDRATION]":     -18.0106,
        "[TRANS_DEHALOGENATION_HF]":-20.0062,
        "[TRANS_ACETYLATION]":      42.0106,
        "[TRANS_SULFONATION]":      79.9568,
        "[TRANS_GLUCURONIDATION]": 176.0321,
        "[TRANS_ISOMERIZATION]":     0.0000,
        "[TRANS_DECARBOXYLATION]": -43.9898,
    })

    # ── PPM threshold ─────────────────────────────────────────────────────
    ppm_threshold: float = 5.0

    @property
    def reaction_tokens(self):
        return list(self.token_to_delta.keys())

    @property
    def special_tokens(self):
        return ["[SEED]"] + self.reaction_tokens

    @cached_property
    def token_to_idx(self) -> Dict[str, int]:
        return {t: i for i, t in enumerate(self.reaction_tokens)}
