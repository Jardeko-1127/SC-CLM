"""V6c tests — inheritance from V6SCLM, append injection, discrete/MLP modes, save/load."""

from __future__ import annotations

import os, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _base() -> str:
    p = Path(__file__).resolve().parents[1] / "models" / "ReactionT5v2-forward"
    return str(p) if p.is_dir() else "sagawa/ReactionT5v2-forward"


def _tok():
    from transformers import T5Tokenizer
    t = T5Tokenizer.from_pretrained(_base())
    t.add_special_tokens({"additional_special_tokens": ["[MASS]"]})
    return t


# ═══════════════════════════════════════════════════════════════════════════
class TestV6cConfig(unittest.TestCase):
    def test_default_discrete(self):
        from src.model.v6c.config import V6cConfig
        c = V6cConfig(); self.assertEqual(c.mass_mode, "discrete"); self.assertFalse(c.is_mlp)

    def test_mlp(self):
        from src.model.v6c.config import V6cConfig
        c = V6cConfig(mass_mode="mlp"); self.assertTrue(c.is_mlp)

    def test_invalid_raises(self):
        from src.model.v6c.config import V6cConfig
        with self.assertRaises(ValueError): V6cConfig(mass_mode="bad")

    def test_inherits_v6a(self):
        from src.model.v6c.config import V6cConfig
        from src.model.v6.config import V6Config
        a, c = V6Config(), V6cConfig()
        self.assertEqual(c.num_mass_bins, a.num_mass_bins)
        self.assertEqual(c.mass_to_bin(15.3), a.mass_to_bin(15.3))


# ═══════════════════════════════════════════════════════════════════════════
class TestV6cDiscrete(unittest.TestCase):
    """V6c in discrete mode — same embedding as V6a, only append differs."""

    _m = _t = None

    @classmethod
    def setUpClass(cls):
        from src.model.v6c.config import V6cConfig
        from src.model.v6c.model import V6cSCLM
        c = V6cConfig(mass_mode="discrete"); c.base_model = _base()
        cls._t = _tok()
        cls._m = V6cSCLM(c)
        cls._m.base_model.resize_token_embeddings(len(cls._t)); cls._m.eval()

    def test_is_v6sclm_subclass(self):
        from src.model.v6.model import V6SCLM
        self.assertIsInstance(self._m, V6SCLM)

    def test_has_mass_embed_not_mlp(self):
        self.assertIsNotNone(self._m.mass_embed)
        self.assertIsNone(self._m.mass_mlp)
        self.assertEqual(self._m.mass_mode, "discrete")

    def test_encode_appends(self):
        tok = self._t
        inp = tok("CCO", max_length=64, padding="max_length", truncation=True, return_tensors="pt")
        d = torch.tensor([2.0])
        cond, _ = self._m._encode(inp["input_ids"], inp["attention_mask"], delta_mz=d)
        base = self._m.base_model.encoder(input_ids=inp["input_ids"],
                                           attention_mask=inp["attention_mask"]).last_hidden_state
        L = base.size(1)
        self.assertTrue(torch.allclose(cond[:, :L, :], base, atol=1e-5),
                        "first L positions = encoder output (mass appended)")
        self.assertEqual(cond[:, L:, :].shape, (1, 1, 768))

    def test_forward(self):
        t = self._t
        i = t("CCO", max_length=64, padding="max_length", truncation=True, return_tensors="pt")
        o = t("CC=O", max_length=64, padding="max_length", truncation=True, return_tensors="pt")
        l = o["input_ids"]; l[l == t.pad_token_id] = -100
        out = self._m(input_ids=i["input_ids"], attention_mask=i["attention_mask"],
                       labels=l, delta_mz=torch.tensor([2.0]))
        self.assertTrue(torch.isfinite(out.loss))

    def test_generate(self):
        t = self._t
        i = t("CCO", max_length=64, padding="max_length", truncation=True, return_tensors="pt")
        ids = self._m.generate(i["input_ids"], i["attention_mask"],
                                delta_mz=torch.tensor([2.0]), max_length=64, num_beams=1)
        self.assertEqual(ids.ndim, 2)

    def test_cfg_generate(self):
        t = self._t
        i = t("CCO", max_length=64, padding="max_length", truncation=True, return_tensors="pt")
        ids = self._m.generate_with_cfg(i["input_ids"], i["attention_mask"],
                                         delta_mz=torch.tensor([2.0]),
                                         guidance_scale=1.5, max_length=64, num_beams=1)
        self.assertEqual(ids.ndim, 2)

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            self._m.save(d, tokenizer=self._t)
            self.assertTrue(Path(d, "v6c_state.pt").exists())
            from src.model.v6c.model import V6cSCLM
            ld = V6cSCLM.load(d); ld.eval()
            self.assertEqual(ld.mass_mode, "discrete")
            self.assertTrue(torch.allclose(self._m.mass_embed.weight, ld.mass_embed.weight))


# ═══════════════════════════════════════════════════════════════════════════
class TestV6cMLP(unittest.TestCase):
    """V6c in MLP mode — continuous embedding, no quantization bins."""

    _m = _t = None

    @classmethod
    def setUpClass(cls):
        from src.model.v6c.config import V6cConfig
        from src.model.v6c.model import V6cSCLM
        c = V6cConfig(mass_mode="mlp"); c.base_model = _base()
        cls._t = _tok()
        cls._m = V6cSCLM(c)
        cls._m.base_model.resize_token_embeddings(len(cls._t)); cls._m.eval()

    def test_has_mlp_not_embed(self):
        self.assertIsNotNone(self._m.mass_mlp)
        self.assertIsNone(self._m.mass_embed)
        self.assertEqual(self._m.mass_mode, "mlp")

    def test_mlp_output_shape(self):
        e = self._m._get_mass_embed(torch.tensor([15.3, -14.7, 0.0]), training=False)
        self.assertEqual(e.shape, (3, 1, 768))

    def test_mlp_different_deltas(self):
        e1 = self._m._get_mass_embed(torch.tensor([15.0]), training=False)
        e2 = self._m._get_mass_embed(torch.tensor([150.0]), training=False)
        self.assertGreater((e1 - e2).abs().sum().item(), 0.01)

    def test_forward_mlp(self):
        t = self._t
        i = t("CCO", max_length=64, padding="max_length", truncation=True, return_tensors="pt")
        o = t("CC=O", max_length=64, padding="max_length", truncation=True, return_tensors="pt")
        l = o["input_ids"]; l[l == t.pad_token_id] = -100
        out = self._m(input_ids=i["input_ids"], attention_mask=i["attention_mask"],
                       labels=l, delta_mz=torch.tensor([2.0]))
        self.assertTrue(torch.isfinite(out.loss))

    def test_cfg_dropout_mlp(self):
        self._m.train()
        d = torch.tensor([15.5] * 32)
        zeros = sum(1 for _ in range(100) for j in range(32)
                    if self._m._get_mass_embed(d, training=True)[j].abs().sum() < 1e-6)
        self.assertGreater(zeros, 0)
        self._m.eval()

    def test_save_load_mlp(self):
        with tempfile.TemporaryDirectory() as d:
            self._m.save(d, tokenizer=self._t)
            self.assertTrue(Path(d, "v6c_state.pt").exists())
            from src.model.v6c.model import V6cSCLM
            ld = V6cSCLM.load(d); ld.eval()
            self.assertEqual(ld.mass_mode, "mlp")
            for (_, op), (_, lp) in zip(self._m.mass_mlp.named_parameters(),
                                         ld.mass_mlp.named_parameters()):
                self.assertTrue(torch.allclose(op, lp))


# ═══════════════════════════════════════════════════════════════════════════
class TestV6cVsV6a(unittest.TestCase):
    """Compare V6a (prepend) vs V6c (append) — same weight init, different cat order."""

    _ma = _mc = _t = None

    @classmethod
    def setUpClass(cls):
        from src.model.v6.config import V6Config
        from src.model.v6.model import V6SCLM
        from src.model.v6c.config import V6cConfig
        from src.model.v6c.model import V6cSCLM
        cls._t = _tok(); b = _base()
        cls._ma = V6SCLM(V6Config()); cls._ma.eval()
        cls._mc = V6cSCLM(V6cConfig(mass_mode="discrete")); cls._mc.eval()

    def test_encode_order_differs(self):
        t = self._t
        i = t("CCO", max_length=64, padding="max_length", truncation=True, return_tensors="pt")
        d = torch.tensor([2.0])
        ca, _ = self._ma._encode(i["input_ids"], i["attention_mask"], delta_mz=d)
        cc, _ = self._mc._encode(i["input_ids"], i["attention_mask"], delta_mz=d)
        self.assertEqual(ca.shape, cc.shape)
        base = self._ma.base_model.encoder(input_ids=i["input_ids"],
                                            attention_mask=i["attention_mask"]).last_hidden_state
        L = base.size(1)
        # V6a: mass at [0], tokens at [1:]
        self.assertTrue(torch.allclose(ca[:, 1:, :], base, atol=1e-5))
        # V6c: tokens at [0:L], mass at [L:]
        self.assertTrue(torch.allclose(cc[:, :L, :], base, atol=1e-5))

    def test_forward_produces_finite_loss(self):
        t = self._t
        i = t("CCO", max_length=64, padding="max_length", truncation=True, return_tensors="pt")
        o = t("CC=O", max_length=64, padding="max_length", truncation=True, return_tensors="pt")
        l = o["input_ids"]; l[l == t.pad_token_id] = -100
        d = torch.tensor([2.0])
        for m in [self._ma, self._mc]:
            with torch.no_grad():
                out = m(input_ids=i["input_ids"], attention_mask=i["attention_mask"],
                         labels=l, delta_mz=d)
            self.assertTrue(torch.isfinite(out.loss))


if __name__ == "__main__":
    unittest.main(verbosity=2)
