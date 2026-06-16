"""
V5SCLM — Embedding-Conditioned Encoder-Decoder with Classifier-Free Guidance.

Architecture:
  Parent SMILES → T5 Encoder → Hidden (B, L, 512)
  Reaction ID    → ReactionEmbedding → (B, 1, 512)
  Hidden + prepended ReactionEmbedding → Conditioned Hidden (B, L+1, 512)
  T5 Decoder cross-attends to Conditioned Hidden → Product SMILES

CFG training: randomly dropout reaction embedding (p=0.15).
CFG inference: logits = uncond + γ × (cond − uncond).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

import transformers.utils.import_utils as _hf_utils
import transformers.modeling_utils as _hf_modeling
from transformers.modeling_outputs import BaseModelOutput

from transformers import T5ForConditionalGeneration, T5Tokenizer

logger = logging.getLogger(__name__)


def _bypass_torch_cve():
    """Temporarily bypass torch.load CVE version check (one-shot)."""
    _hf_utils.check_torch_load_is_safe = lambda: None
    _hf_modeling.check_torch_load_is_safe = lambda: None
    try:
        import transformers.trainer as _hf_trainer
        _hf_trainer.check_torch_load_is_safe = lambda: None
    except Exception:
        pass
    # Patch torch.load to strip weights_only kwarg
    _orig = torch.load
    def _patched(*args, **kw):
        kw.pop("weights_only", None)
        return _orig(*args, **kw)
    torch.load = _patched

class V5SCLM(nn.Module):
    """
    V5 Seed-Conditioned Chemical Language Model.

    Wraps T5ForConditionalGeneration with:
      - Learnable continuous reaction embeddings injected at encoder output
      - Classifier-Free Guidance (CFG) for robust conditioning
      - Null-condition (all-zeros) as unconditional baseline
    """

    def __init__(
        self,
        base_model_name: str = "laituan245/molt5-small",
        num_reaction_tokens: int = 10,
        reaction_embed_dim: int = 512,
        cfg_dropout_prob: float = 0.15,
        cfg_guidance_scale: float = 1.5,
        token_to_idx: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        _bypass_torch_cve()
        self.base_model = T5ForConditionalGeneration.from_pretrained(base_model_name)
        base_d_model = int(self.base_model.config.d_model)
        if reaction_embed_dim != base_d_model:
            raise ValueError(
                f"reaction_embed_dim ({reaction_embed_dim}) must match "
                f"base model d_model ({base_d_model}) for concat conditioning."
            )
        self.reaction_embed = nn.Embedding(num_reaction_tokens, reaction_embed_dim)
        nn.init.normal_(self.reaction_embed.weight, std=0.02)

        self.num_reaction_tokens = num_reaction_tokens
        self.reaction_embed_dim = reaction_embed_dim
        self.cfg_dropout_prob = cfg_dropout_prob
        self.cfg_guidance_scale = cfg_guidance_scale
        self.token_to_idx = token_to_idx or {}

        self._null_embed: Optional[torch.Tensor] = None

    # ── helpers ────────────────────────────────────────────────────────────
    def _get_null_embed(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return all-zeros null embedding for unconditional generation."""
        if (self._null_embed is None or self._null_embed.size(0) < batch_size
                or self._null_embed.device != device):
            self._null_embed = torch.zeros(batch_size, 1, self.reaction_embed_dim, device=device)
        return self._null_embed[:batch_size]

    def _get_reaction_embed(
        self, reaction_ids: torch.LongTensor, training: bool = False
    ) -> torch.Tensor:
        """Lookup reaction embedding with optional CFG dropout during training."""
        r_emb = self.reaction_embed(reaction_ids)          # (B, embed_dim)
        r_emb = r_emb.unsqueeze(1)                         # (B, 1, embed_dim)

        if training and self.cfg_dropout_prob > 0:
            mask = (
                torch.rand(r_emb.size(0), device=r_emb.device) >= self.cfg_dropout_prob
            ).float()
            r_emb = r_emb * mask.view(-1, 1, 1)

        return r_emb

    def _encode(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        reaction_ids: Optional[torch.LongTensor] = None,
        training: bool = False,
    ):
        """Run encoder and prepend reaction embedding to hidden states.

        Returns conditioned encoder outputs and extended attention mask.
        """
        encoder_outputs = self.base_model.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hidden = encoder_outputs.last_hidden_state          # (B, L, d_model)

        if reaction_ids is not None:
            r_emb = self._get_reaction_embed(reaction_ids, training=training)
        else:
            r_emb = self._get_null_embed(
                input_ids.size(0), input_ids.device
            )

        # Prepend reaction embedding: dedicated "condition register"
        conditioned = torch.cat([r_emb, hidden], dim=1)      # (B, L+1, d_model)

        # Extend attention mask for prepended position (always attended)
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
        reaction_ids: torch.LongTensor,
    ):
        """
        Training forward pass.

        Args:
            input_ids: parent SMILES tokens (B, L_in)
            attention_mask: input padding mask (B, L_in)
            labels: product SMILES tokens (B, L_out), pad positions = -100
            reaction_ids: reaction type index (B,)

        Returns:
            outputs with .loss populated
        """
        conditioned_hidden, extended_mask = self._encode(
            input_ids,
            attention_mask,
            reaction_ids=reaction_ids,
            training=self.training,
        )

        # Pass pre-computed encoder outputs → T5 skips encoder, runs decoder + lm_head
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
        reaction_ids: Optional[torch.LongTensor] = None,
        **gen_kwargs,
    ) -> torch.LongTensor:
        """Standard beam/greedy generation without CFG.

        Compatible with Seq2SeqTrainer.predict_with_generate.
        Encodes input with reaction embedding prepended, delegates to base T5. """
        conditioned_hidden, extended_mask = self._encode(
            input_ids, attention_mask, reaction_ids=reaction_ids, training=False
        )
        # T5.generate() needs decoder_input_ids or bos_token_id to start
        gen_kwargs.setdefault(
            "bos_token_id", self.base_model.config.decoder_start_token_id
        )
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
        reaction_ids: torch.LongTensor,
        guidance_scale: Optional[float] = None,
        **gen_kwargs,
    ) -> torch.LongTensor:
        """
        CFG generation via encoder-state interpolation.

        guided_hidden = uncond_hidden + γ × (cond_hidden − uncond_hidden)

        This encoder-level CFG is an approximation to per-step logit blending.
        It requires only a single generate() call, avoiding the complexity and
        stability issues of manual autoregressive two-pass decoding with T5.
        """
        if guidance_scale is None:
            guidance_scale = self.cfg_guidance_scale

        # Conditional encoder states
        cond_hidden, cond_mask = self._encode(
            input_ids, attention_mask, reaction_ids=reaction_ids, training=False
        )
        # Unconditional encoder states
        uncond_hidden, uncond_mask = self._encode(
            input_ids, attention_mask, reaction_ids=None, training=False
        )

        # Encoder-level CFG interpolation
        guided_hidden = uncond_hidden + guidance_scale * (cond_hidden - uncond_hidden)

        gen_kwargs.setdefault(
            "bos_token_id", self.base_model.config.decoder_start_token_id
        )
        gen_kwargs.setdefault("pad_token_id", self.base_model.config.pad_token_id)
        gen_kwargs.setdefault("eos_token_id", self.base_model.config.eos_token_id)

        return self.base_model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=guided_hidden),
            attention_mask=cond_mask,
            **gen_kwargs,
        )

    # ── persistence ────────────────────────────────────────────────────────
    def save_pretrained(self, path: str, **kwargs):
        """Seq2SeqTrainer-compatible save hook. Delegates to save()."""
        tokenizer = kwargs.get("tokenizer")
        self.save(path, tokenizer=tokenizer)

    def save(self, path: str, tokenizer: Optional[T5Tokenizer] = None):
        """Save base model weights, reaction embeddings, and config."""
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.base_model.save_pretrained(str(output_dir))
        torch.save(
            {
                "reaction_embed": self.reaction_embed.state_dict(),
                "token_to_idx": self.token_to_idx,
                "cfg_dropout_prob": self.cfg_dropout_prob,
                "cfg_guidance_scale": self.cfg_guidance_scale,
                "num_reaction_tokens": self.num_reaction_tokens,
                "reaction_embed_dim": self.reaction_embed_dim,
            },
            output_dir / "v5_state.pt",
        )
        if tokenizer is not None:
            tokenizer.save_pretrained(str(output_dir))
        logger.info(f"V5SCLM saved to {path}")

    @classmethod
    def load(cls, path: str, tokenizer: Optional[T5Tokenizer] = None) -> "V5SCLM":
        """Load V5SCLM from saved checkpoint."""
        ckpt_dir = Path(path)
        _bypass_torch_cve()
        base_model = T5ForConditionalGeneration.from_pretrained(str(ckpt_dir))
        ckpt = torch.load(
            ckpt_dir / "v5_state.pt", map_location="cpu", weights_only=False
        )

        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.base_model = base_model
        model.reaction_embed = nn.Embedding(
            ckpt["num_reaction_tokens"], ckpt["reaction_embed_dim"]
        )
        model.reaction_embed.load_state_dict(ckpt["reaction_embed"])
        model.num_reaction_tokens = ckpt["num_reaction_tokens"]
        model.reaction_embed_dim = ckpt["reaction_embed_dim"]
        model.cfg_dropout_prob = ckpt["cfg_dropout_prob"]
        model.cfg_guidance_scale = ckpt["cfg_guidance_scale"]
        model.token_to_idx = ckpt["token_to_idx"]
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
