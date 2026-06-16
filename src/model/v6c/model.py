"""V6cSCLM — inherits V6SCLM, overrides only _encode (append) + _get_mass_embed (MLP branch)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.model.v6.model import V6SCLM, _bypass_torch_cve
from src.model.v6c.config import V6cConfig

logger = logging.getLogger(__name__)


class V6cSCLM(V6SCLM):
    """V6c: V6SCLM subclass with append mass injection + optional MLP.

    Overrides (only 5 methods):
      __init__         — add mass_mlp when mass_mode='mlp'
      _get_mass_embed  — MLP branch or super().discrete
      _encode          — append mass at end (V6a prepends at start)
      save             — v6c_state.pt with mass_mode
      load             — reconstruct from v6c_state.pt
    """

    def __init__(self, config: V6cConfig):
        super().__init__(config)                       # creates self.mass_embed (discrete)
        if config.is_mlp:
            self.mass_mlp = nn.Sequential(
                nn.Linear(1, config.mass_mlp_hidden),
                nn.SiLU(),
                nn.Linear(config.mass_mlp_hidden, config.mass_embed_dim),
            )
            self.mass_embed = None                     # replace discrete embedding
            logger.info("V6cSCLM mass_mode=mlp: MLP(1→%d→%d)", config.mass_mlp_hidden, config.mass_embed_dim)
        else:
            self.mass_mlp = None
            logger.info("V6cSCLM mass_mode=discrete: Embedding(%d, %d)",
                         config.num_mass_bins, config.mass_embed_dim)

    @property
    def mass_mode(self) -> str:
        return self.v6_config.mass_mode

    # ── override: _get_mass_embed (adds MLP branch) ────────────────────────
    def _get_mass_embed(
        self, delta_mz: torch.Tensor, training: bool = False
    ) -> torch.Tensor:
        if self.mass_mode == "mlp":
            m_emb = self.mass_mlp(delta_mz.unsqueeze(-1)).unsqueeze(1)  # (B, 1, 768)
            if training and self.cfg_dropout_prob > 0:
                mask = (torch.rand(m_emb.size(0), device=m_emb.device) >= self.cfg_dropout_prob).float()
                m_emb = m_emb * mask.view(-1, 1, 1)
            return m_emb
        return super()._get_mass_embed(delta_mz, training)  # discrete: V6a path

    # ── override: _encode (APPEND instead of prepend) ──────────────────────
    def _encode(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        delta_mz: Optional[torch.Tensor] = None,
        training: bool = False,
    ):
        encoder_outputs = self.base_model.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hidden = encoder_outputs.last_hidden_state       # (B, L, d_model)

        if delta_mz is not None:
            m_emb = self._get_mass_embed(delta_mz, training=training)
        else:
            m_emb = self._get_null_embed(input_ids.size(0), input_ids.device)

        # V6c: APPEND mass at end (V6a prepends at position 0)
        conditioned = torch.cat([hidden, m_emb], dim=1)  # (B, L+1, d_model)
        ext_mask = torch.cat([
            attention_mask,
            torch.ones(attention_mask.size(0), 1, device=attention_mask.device),
        ], dim=1)
        return conditioned, ext_mask

    # ── override: save (v6c_state.pt with mass_mode) ───────────────────────
    def save(self, path: str, tokenizer: Optional[T5Tokenizer] = None):
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.base_model.save_pretrained(str(output_dir))

        state = {
            "cfg_dropout_prob": self.cfg_dropout_prob,
            "cfg_guidance_scale": self.cfg_guidance_scale,
            "mass_embed_dim": self.mass_embed_dim,
            "mass_mode": self.mass_mode,
            "mass_mlp_hidden": self.v6_config.mass_mlp_hidden,
            "mass_bins": self.v6_config.mass_bins,
        }
        if self.mass_mode == "mlp":
            state["mass_mlp"] = self.mass_mlp.state_dict()
        else:
            state["mass_embed"] = self.mass_embed.state_dict()
            state["num_mass_bins"] = self.v6_config.num_mass_bins

        torch.save(state, output_dir / "v6c_state.pt")
        if tokenizer is not None:
            tokenizer.save_pretrained(str(output_dir))
        logger.info("V6cSCLM saved to %s (mode=%s)", path, self.mass_mode)

    # ── override: load (reconstruct from v6c_state.pt) ─────────────────────
    @classmethod
    def load(cls, path: str, tokenizer: Optional[T5Tokenizer] = None) -> "V6cSCLM":
        ckpt_dir = Path(path)
        _bypass_torch_cve()
        base_model = T5ForConditionalGeneration.from_pretrained(str(ckpt_dir))
        ckpt = torch.load(ckpt_dir / "v6c_state.pt", map_location="cpu", weights_only=False)

        mass_mode = ckpt.get("mass_mode", "discrete")
        mass_mlp_hidden = ckpt.get("mass_mlp_hidden", 256)
        config = V6cConfig(
            mass_mode=mass_mode,
            mass_mlp_hidden=mass_mlp_hidden,
            mass_bins=ckpt.get("mass_bins", V6cConfig().mass_bins),
        )

        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.v6_config = config
        model.base_model = base_model
        model.mass_embed_dim = ckpt["mass_embed_dim"]
        model.cfg_dropout_prob = ckpt["cfg_dropout_prob"]
        model.cfg_guidance_scale = ckpt["cfg_guidance_scale"]

        if mass_mode == "mlp":
            model.mass_mlp = nn.Sequential(
                nn.Linear(1, mass_mlp_hidden), nn.SiLU(), nn.Linear(mass_mlp_hidden, model.mass_embed_dim),
            )
            model.mass_mlp.load_state_dict(ckpt["mass_mlp"])
            model.mass_embed = None
        else:
            model.mass_embed = nn.Embedding(ckpt["num_mass_bins"], ckpt["mass_embed_dim"])
            model.mass_embed.load_state_dict(ckpt["mass_embed"])
            model.mass_mlp = None

        model._null_embed = None
        return model
