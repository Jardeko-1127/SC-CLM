"""V6c configuration — extends V6Config with mass injection mode."""

from dataclasses import dataclass

from src.model.v6.config import V6Config


@dataclass
class V6cConfig(V6Config):
    """V6c config: V6a + mass_mode (discrete bins or continuous MLP).

    mass_mode='discrete': 532-bin nn.Embedding (same as V6a, control baseline)
    mass_mode='mlp':       MLP(1→256→768) continuous embedding (no quantization error)
    """

    mass_mode: str = "discrete"  # "discrete" or "mlp"
    mass_mlp_hidden: int = 256

    def __post_init__(self):
        if self.mass_mode not in ("discrete", "mlp"):
            raise ValueError(f"mass_mode must be 'discrete' or 'mlp', got {self.mass_mode!r}")

    @property
    def is_mlp(self) -> bool:
        return self.mass_mode == "mlp"
