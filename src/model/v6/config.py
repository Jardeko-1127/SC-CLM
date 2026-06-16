"""V6a configuration — continuous mass-delta conditioning replacing discrete reaction tokens."""

from dataclasses import dataclass, field
from typing import List, Tuple

import torch


def _build_mass_bins() -> List[Tuple[float, float, float]]:
    """Pre-compute mass bin definitions as (left_edge, right_edge, center).

    Returns list of (left, right, center) tuples sorted by left edge.
    Total: 532 bins covering core, outer, and extreme regions.
    """
    bins = []

    # Core [-100, 200) @ 1 Da → 300 bins, center = left + 0.5
    for left in range(-100, 200):
        bins.append((float(left), float(left + 1), left + 0.5))

    # Outer [-500, -100) @ 5 Da → 80 bins, center = left + 2.5
    for left in range(-500, -100, 5):
        bins.append((float(left), float(left + 5), left + 2.5))

    # Outer [200, 500) @ 5 Da → 60 bins, center = left + 2.5
    for left in range(200, 500, 5):
        bins.append((float(left), float(left + 5), left + 2.5))

    # Extreme (< -500) @ 10 Da → 57 bins, center = left + 5
    for left in range(-1070, -500, 10):
        bins.append((float(left), float(left + 10), left + 5))

    # Extreme (>= 500) @ 10 Da → 35 bins, center = left + 5
    for left in range(500, 850, 10):
        bins.append((float(left), float(left + 10), left + 5))

    bins.sort(key=lambda x: x[0])
    return bins


@dataclass
class V6Config:
    """Centralized configuration for V6a mass-conditioned CLM."""

    # ── Model ──────────────────────────────────────────────────────────────
    base_model: str = "sagawa/ReactionT5v2-forward"  # or local models/ReactionT5v2-forward
    mass_embed_dim: int = 768  # must match ReactionT5 d_model

    # ── Mass quantization ──────────────────────────────────────────────────
    mass_bins: List[Tuple[float, float, float]] = field(default_factory=_build_mass_bins)

    # ── LoRA (V5B1 style) ──────────────────────────────────────────────────
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q", "k", "v", "o"])

    # ── CFG ────────────────────────────────────────────────────────────────
    cfg_dropout_prob: float = 0.15
    cfg_guidance_scale: float = 1.5

    # ── Training ───────────────────────────────────────────────────────────
    micro_batch: int = 1
    grad_accum: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    num_epochs: int = 30
    max_input_len: int = 256
    max_output_len: int = 128
    seed: int = 42
    patience: int = 10
    metric_for_best: str = "ppm_pass_rate"
    greater_is_better: bool = True

    # ── Paths ──────────────────────────────────────────────────────────────
    train_csv: str = "data/processed/v6_train.csv"
    val_csv: str = "data/processed/v6_val.csv"
    output_dir: str = "results/checkpoints/v6a"
    log_dir: str = "logs/v6a"

    # ── PPM ────────────────────────────────────────────────────────────────
    ppm_threshold: float = 5.0

    @property
    def num_mass_bins(self) -> int:
        return len(self.mass_bins)

    def mass_to_bin(self, delta_mz: float) -> int:
        """Quantize delta_mz to nearest bin index.

        For values outside all bins, clamp to nearest bin.
        """
        bins = self.mass_bins
        # Binary search through bin edges
        lo, hi = 0, len(bins) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            left, right, _center = bins[mid]
            if delta_mz < left:
                hi = mid - 1
            elif delta_mz >= right:
                lo = mid + 1
            else:
                return mid
        # Clamp to nearest boundary bin
        if lo >= len(bins):
            return len(bins) - 1
        return max(0, lo)

    def mass_to_bin_tensor(self, delta_mz: torch.Tensor) -> torch.LongTensor:
        """Vectorized quantization — delegates to scalar mass_to_bin for consistency."""
        vals = delta_mz.detach().cpu().float().tolist()
        indices = [self.mass_to_bin(float(v)) for v in (vals if isinstance(vals, list) else [vals])]
        return torch.tensor(indices, dtype=torch.long)

    def bin_to_mass(self, bin_idx: int) -> float:
        """Return the center delta_mz value for a bin index."""
        return self.mass_bins[bin_idx][2]
