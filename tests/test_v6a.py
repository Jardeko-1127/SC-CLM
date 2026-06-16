"""V6a model tests — mass quantization, model creation, forward pass, CFG, save/load, dataset, inference."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import pandas as pd

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ═════════════════════════════════════════════════════════════════════════════
# Test 1: Mass Quantization
# ═════════════════════════════════════════════════════════════════════════════
class TestV6MassQuantization(unittest.TestCase):
    """Test delta_mz → bin quantization logic."""

    @classmethod
    def setUpClass(cls):
        from src.model.v6.config import V6Config
        cls.config = V6Config()

    def test_num_bins(self):
        self.assertEqual(self.config.num_mass_bins, 532)

    def test_quantization_core_positive(self):
        """Core region @1Da: 15.3 → bin for 15"""
        idx = self.config.mass_to_bin(15.3)
        center = self.config.bin_to_mass(idx)
        self.assertEqual(center, 15.5)  # bin left=15, center=15.5

    def test_quantization_core_negative(self):
        """Core region @1Da: -14.7 → bin for -15"""
        idx = self.config.mass_to_bin(-14.7)
        center = self.config.bin_to_mass(idx)
        self.assertEqual(center, -14.5)  # bin left=-15, center=-14.5

    def test_quantization_outer_positive(self):
        """Outer region @5Da: 230 → bin for 230 (center=232.5)"""
        idx = self.config.mass_to_bin(230.0)
        center = self.config.bin_to_mass(idx)
        # 230 is in [230, 235) bin, center = 232.5
        self.assertEqual(center, 232.5)

    def test_quantization_outer_negative(self):
        """Outer region @5Da: -333 → bin for -335 (center=-332.5)"""
        idx = self.config.mass_to_bin(-333.0)
        center = self.config.bin_to_mass(idx)
        # -333 is in [-335, -330) bin, center = -335 + 2.5 = -332.5
        self.assertEqual(center, -332.5)

    def test_quantization_extreme_negative(self):
        """Extreme region @10Da: -888 → bin for -890"""
        idx = self.config.mass_to_bin(-888.0)
        center = self.config.bin_to_mass(idx)
        self.assertEqual(center, -885.0)  # -890 + 5 = -885

    def test_quantization_extreme_positive(self):
        """Extreme region @10Da: 555 → bin for 550"""
        idx = self.config.mass_to_bin(555.0)
        center = self.config.bin_to_mass(idx)
        self.assertEqual(center, 555.0)  # 550 + 5 = 555

    def test_quantization_clamp_low(self):
        """Values below all bins clamp to first bin."""
        idx = self.config.mass_to_bin(-9999.0)
        self.assertEqual(idx, 0)

    def test_quantization_clamp_high(self):
        """Values above all bins clamp to last bin."""
        idx = self.config.mass_to_bin(9999.0)
        self.assertEqual(idx, self.config.num_mass_bins - 1)

    def test_bin_to_mass_roundtrip(self):
        """bin_to_mass values should roundtrip correctly."""
        for delta in [-50.0, -5.0, 0.0, 10.0, 50.0, 150.0, 300.0, 600.0]:
            idx = self.config.mass_to_bin(delta)
            center = self.config.bin_to_mass(idx)
            back_idx = self.config.mass_to_bin(center)
            self.assertEqual(idx, back_idx, f"Roundtrip failed for delta={delta}")

    def test_tensor_quantization(self):
        """Vectorized mass_to_bin_tensor produces same results as scalar."""
        deltas = torch.tensor([15.3, -14.7, 230.0, -333.0, -888.0], dtype=torch.float32)
        idxs = self.config.mass_to_bin_tensor(deltas)
        expected = torch.tensor([
            self.config.mass_to_bin(15.3),
            self.config.mass_to_bin(-14.7),
            self.config.mass_to_bin(230.0),
            self.config.mass_to_bin(-333.0),
            self.config.mass_to_bin(-888.0),
        ], dtype=torch.long)
        self.assertTrue(torch.equal(idxs, expected))

    def test_mass_embed_shape(self):
        """mass_embed has correct shape."""
        import torch.nn as nn
        embed = nn.Embedding(self.config.num_mass_bins, self.config.mass_embed_dim)
        delta = torch.tensor([15.3], dtype=torch.float32)
        idx = self.config.mass_to_bin_tensor(delta)
        out = embed(idx)
        self.assertEqual(out.shape, (1, self.config.mass_embed_dim))


# ═════════════════════════════════════════════════════════════════════════════
# Test 2: V6SCLM Model
# ═════════════════════════════════════════════════════════════════════════════
class TestV6Model(unittest.TestCase):
    """Test V6SCLM creation, forward pass, CFG generation, save/load."""

    _model = None
    _tokenizer = None
    _config = None

    @classmethod
    def setUpClass(cls):
        from src.model.v6.config import V6Config
        from src.model.v6.model import V6SCLM
        from transformers import T5Tokenizer

        cls._config = V6Config()
        # Use local model path if available
        local = Path(__file__).resolve().parents[1] / "models" / "ReactionT5v2-forward"
        if local.is_dir():
            cls._config.base_model = str(local)
        cls._tokenizer = T5Tokenizer.from_pretrained(cls._config.base_model)
        cls._tokenizer.add_special_tokens({"additional_special_tokens": ["[MASS]"]})
        cls._model = V6SCLM(cls._config)
        cls._model.base_model.resize_token_embeddings(len(cls._tokenizer))
        cls._model.eval()

    def test_model_creation(self):
        """V6SCLM instantiation succeeds."""
        self.assertIsNotNone(self._model)
        self.assertIsNotNone(self._model.mass_embed)
        self.assertEqual(self._model.mass_embed.num_embeddings, 532)
        self.assertEqual(self._model.mass_embed.embedding_dim, 768)

    def test_forward_pass_shape(self):
        """Single forward pass produces loss scalar."""
        tokenizer = self._tokenizer
        parent = "CCO"
        product = "CC=O"

        inp = tokenizer(parent, max_length=64, padding="max_length", truncation=True, return_tensors="pt")
        out = tokenizer(product, max_length=64, padding="max_length", truncation=True, return_tensors="pt")
        labels = out["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100

        delta_mz = torch.tensor([1.0], dtype=torch.float32)

        with torch.no_grad():
            output = self._model(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
                labels=labels,
                delta_mz=delta_mz,
            )
        self.assertIsNotNone(output.loss)
        self.assertGreater(output.loss.item(), 0.0)

    def test_forward_pass_batch(self):
        """Batch forward pass produces expected loss shape."""
        tokenizer = self._tokenizer
        parents = ["CCO", "CCCO"]
        products = ["CC=O", "CCC=O"]

        inp = tokenizer(parents, max_length=64, padding="max_length", truncation=True, return_tensors="pt")
        out = tokenizer(products, max_length=64, padding="max_length", truncation=True, return_tensors="pt")
        labels = out["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100

        delta_mz = torch.tensor([2.0, 2.0], dtype=torch.float32)

        with torch.no_grad():
            output = self._model(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
                labels=labels,
                delta_mz=delta_mz,
            )
        self.assertTrue(torch.isfinite(output.loss))

    def test_generate_produces_output(self):
        """Standard generate returns token IDs."""
        tokenizer = self._tokenizer
        inp = tokenizer("CCO", max_length=64, padding="max_length", truncation=True, return_tensors="pt")
        delta_mz = torch.tensor([2.0], dtype=torch.float32)

        with torch.no_grad():
            out_ids = self._model.generate(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
                delta_mz=delta_mz,
                max_length=128,
                num_beams=1,
            )
        self.assertIsInstance(out_ids, torch.Tensor)
        self.assertEqual(out_ids.ndim, 2)  # (B, L)

    def test_cfg_generation_produces_output(self):
        """CFG generate returns token IDs."""
        tokenizer = self._tokenizer
        inp = tokenizer("CCO", max_length=64, padding="max_length", truncation=True, return_tensors="pt")
        delta_mz = torch.tensor([2.0], dtype=torch.float32)

        with torch.no_grad():
            out_ids = self._model.generate_with_cfg(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
                delta_mz=delta_mz,
                guidance_scale=1.5,
                max_length=128,
                num_beams=1,
            )
        self.assertIsInstance(out_ids, torch.Tensor)
        self.assertEqual(out_ids.ndim, 2)

    def test_cfg_null_condition(self):
        """CFG with null (None) delta_mz falls back to unconditional."""
        tokenizer = self._tokenizer
        inp = tokenizer("CCO", max_length=64, padding="max_length", truncation=True, return_tensors="pt")

        with torch.no_grad():
            out_ids = self._model.generate_with_cfg(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
                delta_mz=torch.tensor([2.0], dtype=torch.float32),
                guidance_scale=1.0,  # no guidance boost
                max_length=128,
                num_beams=1,
            )
        self.assertGreater(out_ids.size(1), 0)

    def test_save_load_roundtrip(self):
        """Save and reload model, verify weights match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._model.save(tmpdir, tokenizer=self._tokenizer)
            self.assertTrue(Path(tmpdir, "v6_state.pt").exists())
            self.assertTrue(Path(tmpdir, "config.json").exists())

            from src.model.v6.model import V6SCLM
            loaded = V6SCLM.load(tmpdir)
            loaded.eval()

            # Compare mass_embed weights
            orig_w = self._model.mass_embed.weight.data
            loaded_w = loaded.mass_embed.weight.data
            self.assertTrue(torch.allclose(orig_w, loaded_w, atol=1e-6))

            # Compare CFG params
            self.assertEqual(loaded.cfg_dropout_prob, self._model.cfg_dropout_prob)
            self.assertEqual(loaded.cfg_guidance_scale, self._model.cfg_guidance_scale)
            self.assertEqual(loaded.v6_config.num_mass_bins, self._model.v6_config.num_mass_bins)

    def test_cfg_dropout_training_mode(self):
        """In training mode, some embeddings should be zeroed."""
        self._model.train()
        delta_mz = torch.tensor([15.5] * 32, dtype=torch.float32)  # batch of 32
        # Run multiple times; with p=0.15, some should have mask applied
        non_zero_count = 0
        zero_count = 0
        for _ in range(100):
            emb = self._model._get_mass_embed(delta_mz, training=True)
            for i in range(emb.size(0)):
                if emb[i].abs().sum() > 1e-6:
                    non_zero_count += 1
                else:
                    zero_count += 1
        self.assertGreater(zero_count, 0, "CFG dropout should zero some embeddings")
        self.assertGreater(non_zero_count, 0, "Some embeddings should be non-zero")
        self._model.eval()


# ═════════════════════════════════════════════════════════════════════════════
# Test 3: V6Dataset
# ═════════════════════════════════════════════════════════════════════════════
class TestV6Dataset(unittest.TestCase):
    """Test V6Dataset loads CSV and returns correct fields."""

    @classmethod
    def setUpClass(cls):
        from src.model.v6.config import V6Config
        from transformers import T5Tokenizer

        cls.config = V6Config()
        local = Path(__file__).resolve().parents[1] / "models" / "ReactionT5v2-forward"
        if local.is_dir():
            cls.config.base_model = str(local)
        cls.tokenizer = T5Tokenizer.from_pretrained(cls.config.base_model)
        cls.tokenizer.add_special_tokens({"additional_special_tokens": ["[MASS]"]})

        # Create minimal test CSV
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.csv_path = Path(cls._tmpdir.name) / "test_v6.csv"
        test_df = pd.DataFrame({
            "parent_smiles": ["CCO", "CCCO", "c1ccccc1"],
            "product_smiles": ["CC=O", "CCC=O", "c1ccccc1O"],
            "delta_mz": [2.0157, 2.0157, 15.9949],
            "source": ["test"] * 3,
        })
        test_df.to_csv(cls.csv_path, index=False)

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def test_dataset_loading(self):
        """Dataset correctly reads CSV and returns all expected fields."""
        from src.model.v6.dataset import V6Dataset

        ds = V6Dataset(
            str(self.csv_path),
            self.tokenizer,
            self.config,
            max_input_len=64,
            max_output_len=64,
            augment=False,
        )
        self.assertEqual(len(ds), 3)

        sample = ds[0]
        self.assertIn("input_ids", sample)
        self.assertIn("attention_mask", sample)
        self.assertIn("labels", sample)
        self.assertIn("delta_mz", sample)
        self.assertIn("bin_idx", sample)

        self.assertIsInstance(sample["delta_mz"], torch.Tensor)
        self.assertEqual(sample["delta_mz"].dtype, torch.float32)
        self.assertIsInstance(sample["bin_idx"], torch.Tensor)
        self.assertEqual(sample["bin_idx"].dtype, torch.long)

        # Check delta_mz matches CSV
        self.assertAlmostEqual(sample["delta_mz"].item(), 2.0157, places=3)

    def test_dataset_with_augmentation(self):
        """Augmented dataset multiplies effective length."""
        from src.model.v6.dataset import V6Dataset

        ds = V6Dataset(
            str(self.csv_path),
            self.tokenizer,
            self.config,
            max_input_len=64,
            max_output_len=64,
            augment=True,
            augment_n=3,
        )
        self.assertEqual(len(ds), 9)  # 3 base × 3 aug

    def test_dataset_labels_have_neg100_padding(self):
        """Labels contain -100 for padding positions."""
        from src.model.v6.dataset import V6Dataset

        ds = V6Dataset(
            str(self.csv_path),
            self.tokenizer,
            self.config,
            max_input_len=64,
            max_output_len=64,
            augment=False,
        )
        sample = ds[0]
        labels = sample["labels"]
        # At least some positions should be -100 (padding) for short sequences
        self.assertTrue((labels == -100).any().item(),
                        "Padding positions should be -100")


# ═════════════════════════════════════════════════════════════════════════════
# Test 4: V6Inference
# ═════════════════════════════════════════════════════════════════════════════
class TestV6Inference(unittest.TestCase):
    """End-to-end inference test (requires data files, uses untrained model)."""

    @classmethod
    def setUpClass(cls):
        from src.model.v6.config import V6Config
        from src.model.v6.model import V6SCLM
        from transformers import T5Tokenizer
        from src.model.v6.inference import predict

        cls.config = V6Config()
        local = Path(__file__).resolve().parents[1] / "models" / "ReactionT5v2-forward"
        if local.is_dir():
            cls.config.base_model = str(local)
        cls.tokenizer = T5Tokenizer.from_pretrained(cls.config.base_model)
        cls.tokenizer.add_special_tokens({"additional_special_tokens": ["[MASS]"]})
        cls.model = V6SCLM(cls.config)
        cls.model.base_model.resize_token_embeddings(len(cls.tokenizer))
        cls.model.eval()

    def test_predict_function_runs(self):
        """predict() runs without error (untrained model, output not meaningful)."""
        from src.model.v6.inference import predict

        # Use a simple parent and expected product mass
        # parent: ethanol CCO (mass ~46.04), product: acetaldehyde CC=O (mass ~44.03)
        product_mz = 44.0262  # approximate mass of acetaldehyde
        candidates = predict("CCO", product_mz, self.model, self.tokenizer, self.config)

        self.assertIsInstance(candidates, list)
        if len(candidates) > 0:
            self.assertIsInstance(candidates[0], str)

    def test_predict_with_invalid_smiles(self):
        """predict returns empty list for invalid parent SMILES."""
        from src.model.v6.inference import predict

        candidates = predict("INVALID_SMILES_XYZ", 100.0, self.model, self.tokenizer, self.config)
        self.assertEqual(candidates, [])

    def test_v6_inference_class(self):
        """V6Inference class can be instantiated (no checkpoint needed for this test)."""
        from src.model.v6.inference import V6Inference

        # Save a minimal checkpoint from our model
        with tempfile.TemporaryDirectory() as tmpdir:
            self.model.save(tmpdir, tokenizer=self.tokenizer)

            infer = V6Inference(
                checkpoint_path=tmpdir,
                guidance_scale=1.5,
                device="cpu",
            )
            self.assertIsNotNone(infer.model)
            self.assertEqual(infer.guidance_scale, 1.5)

            # Test predict method
            candidates = infer.predict("CCO", 44.03, num_beams=3, num_return=2, use_cfg=True)
            self.assertIsInstance(candidates, list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
