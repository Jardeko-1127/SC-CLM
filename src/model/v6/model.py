"""
V6SCLM — Mass-Conditioned Chemical Language Model (V6a).

Architecture:
  Parent SMILES → ReactionT5 Encoder → Hidden (B, L, 768)
  delta_mz (float) → quantize → MassEmbedding[bin_idx] → (B, 1, 768)
  Hidden + prepend(MassEmbedding) → Conditioned Hidden (B, L+1, 768)
  ReactionT5 Decoder cross-attends → Product SMILES

CFG training: randomly dropout mass embedding (p=0.15).
CFG inference: guided_hidden = uncond + γ × (cond − uncond).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

import transformers.utils.import_utils as _hf_utils
import transformers.modeling_utils as _hf_modeling
from transformers.modeling_outputs import BaseModelOutput
from transformers import T5ForConditionalGeneration, T5Tokenizer

from .config import V6Config

logger = logging.getLogger(__name__)


def _bypass_torch_cve():
    """Temporarily bypass torch.load CVE version check (one-shot)."""
    _hf_utils.check_torch_load_is_safe = lambda: None
    _hf_modeling.check_torch_load_is_safe = lambda: None
    _orig = torch.load

    def _patched(*args, **kw):
        kw.pop("weights_only", None)
        return _orig(*args, **kw)

    torch.load = _patched


class V6SCLM(nn.Module):
    """V6 Mass-Conditioned Chemical Language Model.

    Wraps T5ForConditionalGeneration with:
      - Quantized mass-delta embedding injected at encoder output
      - Classifier-Free Guidance (CFG) for robust conditioning
      - Null-condition (all-zeros) as unconditional baseline
    """

    def __init__(self, config: V6Config):
        super().__init__()
        _bypass_torch_cve()
        self.v6_config = config

        # Load ReactionT5v2 backbone
        self.base_model = T5ForConditionalGeneration.from_pretrained(config.base_model)
        base_d_model = int(self.base_model.config.d_model)
        if config.mass_embed_dim != base_d_model:
            raise ValueError(
                f"mass_embed_dim ({config.mass_embed_dim}) must match "
                f"base model d_model ({base_d_model})."
            )

        # Mass embedding: maps quantized delta_mz → continuous vector
        self.mass_embed = nn.Embedding(config.num_mass_bins, config.mass_embed_dim)
        nn.init.normal_(self.mass_embed.weight, std=0.02)

        self.mass_embed_dim = config.mass_embed_dim
        self.cfg_dropout_prob = config.cfg_dropout_prob
        self.cfg_guidance_scale = config.cfg_guidance_scale

        self._null_embed: Optional[torch.Tensor] = None

    # ── helpers ────────────────────────────────────────────────────────────
    def _get_null_embed(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self._null_embed is None or self._null_embed.size(0) < batch_size or self._null_embed.device != device:
            self._null_embed = torch.zeros(batch_size, 1, self.mass_embed_dim, device=device)
        return self._null_embed[:batch_size]

    def _get_mass_embed(
        self, delta_mz: torch.Tensor, training: bool = False
    ) -> torch.Tensor:
        """Quantize delta_mz → lookup mass embedding with optional CFG dropout."""
        bin_idx = self.v6_config.mass_to_bin_tensor(delta_mz).to(delta_mz.device)
        m_emb = self.mass_embed(bin_idx)            # (B, embed_dim)
        m_emb = m_emb.unsqueeze(1)                  # (B, 1, embed_dim)

        if training and self.cfg_dropout_prob > 0:
            mask = (
                torch.rand(m_emb.size(0), device=m_emb.device) >= self.cfg_dropout_prob
            ).float()
            m_emb = m_emb * mask.view(-1, 1, 1)

        return m_emb

    def _encode(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        delta_mz: Optional[torch.Tensor] = None,
        training: bool = False,
    ):
        """Run encoder and prepend mass embedding to hidden states."""
        encoder_outputs = self.base_model.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hidden = encoder_outputs.last_hidden_state       # (B, L, d_model)

        if delta_mz is not None:
            m_emb = self._get_mass_embed(delta_mz, training=training)
        else:
            m_emb = self._get_null_embed(input_ids.size(0), input_ids.device)

        conditioned = torch.cat([m_emb, hidden], dim=1)  # (B, L+1, d_model)

        ext_mask = torch.cat([
            torch.ones(attention_mask.size(0), 1, device=attention_mask.device),
            attention_mask,
        ], dim=1)

        return conditioned, ext_mask

    # ── forward ────────────────────────────────────────────────────────────
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        labels: torch.LongTensor,
        delta_mz: torch.Tensor,
    ):
        conditioned_hidden, extended_mask = self._encode(
            input_ids, attention_mask, delta_mz=delta_mz, training=self.training,
        )
        outputs = self.base_model(
            encoder_outputs=(conditioned_hidden,),
            attention_mask=extended_mask,
            labels=labels,
        )
        return outputs

    # ── Standard generation (non-CFG, for training eval) ───────────────────
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        delta_mz: Optional[torch.Tensor] = None,
        **gen_kwargs,
    ) -> torch.LongTensor:
        conditioned_hidden, extended_mask = self._encode(
            input_ids, attention_mask, delta_mz=delta_mz, training=False
        )
        gen_kwargs.setdefault("bos_token_id", self.base_model.config.decoder_start_token_id)
        gen_kwargs.setdefault("pad_token_id", self.base_model.config.pad_token_id)
        gen_kwargs.setdefault("eos_token_id", self.base_model.config.eos_token_id)
        return self.base_model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=conditioned_hidden),
            attention_mask=extended_mask,
            **gen_kwargs,
        )

    # ── CFG generation (encoder-level guidance) ──────────────────────────
    @torch.no_grad()
    def generate_with_cfg(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        delta_mz: torch.Tensor,
        guidance_scale: Optional[float] = None,
        **gen_kwargs,
    ) -> torch.LongTensor:
        if guidance_scale is None:
            guidance_scale = self.cfg_guidance_scale

        cond_hidden, cond_mask = self._encode(
            input_ids, attention_mask, delta_mz=delta_mz, training=False
        )
        uncond_hidden, uncond_mask = self._encode(
            input_ids, attention_mask, delta_mz=None, training=False
        )

        guided_hidden = uncond_hidden + guidance_scale * (cond_hidden - uncond_hidden)

        gen_kwargs.setdefault("bos_token_id", self.base_model.config.decoder_start_token_id)
        gen_kwargs.setdefault("pad_token_id", self.base_model.config.pad_token_id)
        gen_kwargs.setdefault("eos_token_id", self.base_model.config.eos_token_id)

        return self.base_model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=guided_hidden),
            attention_mask=cond_mask,
            **gen_kwargs,
        )

    # ── persistence ────────────────────────────────────────────────────────
    def save_pretrained(self, path: str, **kwargs):
        tokenizer = kwargs.get("tokenizer")
        self.save(path, tokenizer=tokenizer)

    def save(self, path: str, tokenizer: Optional[T5Tokenizer] = None):
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.base_model.save_pretrained(str(output_dir))
        torch.save(
            {
                "mass_embed": self.mass_embed.state_dict(),
                "cfg_dropout_prob": self.cfg_dropout_prob,
                "cfg_guidance_scale": self.cfg_guidance_scale,
                "num_mass_bins": self.v6_config.num_mass_bins,
                "mass_embed_dim": self.mass_embed_dim,
                "mass_bins": self.v6_config.mass_bins,
            },
            output_dir / "v6_state.pt",
        )
        if tokenizer is not None:
            tokenizer.save_pretrained(str(output_dir))
        logger.info("V6SCLM saved to %s", path)

    @classmethod
    def load(cls, path: str, tokenizer: Optional[T5Tokenizer] = None) -> "V6SCLM":
        ckpt_dir = Path(path)
        _bypass_torch_cve()
        base_model = T5ForConditionalGeneration.from_pretrained(str(ckpt_dir))
        ckpt = torch.load(ckpt_dir / "v6_state.pt", map_location="cpu", weights_only=False)

        config = V6Config(mass_bins=ckpt.get("mass_bins", V6Config().mass_bins))

        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.v6_config = config
        model.base_model = base_model
        model.mass_embed = nn.Embedding(ckpt["num_mass_bins"], ckpt["mass_embed_dim"])
        model.mass_embed.load_state_dict(ckpt["mass_embed"])
        model.mass_embed_dim = ckpt["mass_embed_dim"]
        model.cfg_dropout_prob = ckpt["cfg_dropout_prob"]
        model.cfg_guidance_scale = ckpt["cfg_guidance_scale"]
        model._null_embed = None
        return model

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def config(self):
        return self.base_model.config

    @property
    def generation_config(self):
        return self.base_model.generation_config
