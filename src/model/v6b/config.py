"""V6b configuration — OpenNMT Transformer with self-contained mass quantization.

Mass bins are constructed identically to V6a (532 bins, same edges/centers)
but defined here to avoid the PyTorch import chain triggered by src.model.v6.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


def _build_mass_bins() -> List[Tuple[float, float, float]]:
    """Identical to V6Config._build_mass_bins — 532 bins across core/outer/extreme."""
    bins = []
    # Core [-100, 200) @ 1 Da
    for left in range(-100, 200):
        bins.append((float(left), float(left + 1), left + 0.5))
    # Outer [-500, -100) @ 5 Da
    for left in range(-500, -100, 5):
        bins.append((float(left), float(left + 5), left + 2.5))
    # Outer [200, 500) @ 5 Da
    for left in range(200, 500, 5):
        bins.append((float(left), float(left + 5), left + 2.5))
    # Extreme (< -500) @ 10 Da
    for left in range(-1070, -500, 10):
        bins.append((float(left), float(left + 10), left + 5))
    # Extreme (>= 500) @ 10 Da
    for left in range(500, 850, 10):
        bins.append((float(left), float(left + 10), left + 5))
    bins.sort(key=lambda x: x[0])
    return bins


@dataclass
class V6bConfig:
    """V6b MetaReact-inspired OpenNMT Transformer configuration.

    Mass quantization is self-contained (no torch dependency) for compatibility
    with OpenNMT's torch version requirements.
    """

    # ── Mass quantization (same bins as V6a) ──────────────────────────────
    mass_bins: List[Tuple[float, float, float]] = field(default_factory=_build_mass_bins)
    mass_bins_count: int = 532
    mass_token_offset: int = 1000

    # ── Model architecture ─────────────────────────────────────────────────
    num_layers: int = 6
    num_heads: int = 8
    hidden_size: int = 512
    word_vec_size: int = 512
    ffn_size: int = 2048
    dropout: float = 0.1

    # ── Training ───────────────────────────────────────────────────────────
    batch_size: int = 4096
    batch_type: str = "tokens"
    accum_count: int = 4
    learning_rate: float = 2.0
    warmup_steps: int = 8000
    train_steps: int = 50000
    valid_steps: int = 2000
    save_checkpoint_steps: int = 2000
    keep_checkpoint: int = 5
    seed: int = 42

    # ── Sequence lengths ──────────────────────────────────────────────────
    max_src_len: int = 500
    max_tgt_len: int = 500

    # ── Inference ─────────────────────────────────────────────────────────
    beam_size: int = 20
    n_best: int = 20
    topk: int = 5

    # ── Paths ──────────────────────────────────────────────────────────────
    data_dir: str = "data/processed/v6b"
    model_dir: str = "results/checkpoints/v6b"
    log_dir: str = "logs/v6b"

    @property
    def num_mass_bins(self) -> int:
        return len(self.mass_bins)

    def mass_to_bin(self, delta_mz: float) -> int:
        """Quantize delta_mz to nearest bin index (binary search)."""
        bins = self.mass_bins
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
        if lo >= len(bins):
            return len(bins) - 1
        return max(0, lo)

    def bin_to_mass(self, bin_idx: int) -> float:
        """Return center delta_mz for bin index."""
        return self.mass_bins[bin_idx][2]

    def mass_token(self, delta_mz: float) -> str:
        """Encode delta_mz as MASS_{bin_idx} token string (bin_idx is unique regardless of sign)."""
        idx = self.mass_to_bin(delta_mz)
        return f"MASS_{idx}"

    @property
    def data_prefix(self) -> str:
        return str(Path(self.data_dir) / "v6b")
