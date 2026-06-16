"""V6b tests — data preparation, mass token encoding, OpenNMT format, inference interface."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Import functions directly (module level, not class level)
from src.model.v6b.config import V6bConfig
from src.model.v6b.prepare_data import _kekulize, _space_chars
from src.model.v6b.inference import format_input

REPO_ROOT = Path(__file__).resolve().parents[1]


# ═════════════════════════════════════════════════════════════════════════════
# Test 1: V6bConfig
# ═════════════════════════════════════════════════════════════════════════════
class TestV6bConfig(unittest.TestCase):
    """Test V6bConfig mass token encoding and V6a config sharing."""

    @classmethod
    def setUpClass(cls):
        cls.config = V6bConfig()

    def test_mass_bins_match_v6a(self):
        from src.model.v6.config import V6Config
        v6a = V6Config()
        self.assertEqual(self.config.mass_bins_count, v6a.num_mass_bins)
        self.assertEqual(len(self.config.mass_bins), len(v6a.mass_bins))

    def test_mass_to_bin_shared(self):
        idx = self.config.mass_to_bin(15.3)
        from src.model.v6.config import V6Config
        v6a = V6Config()
        self.assertEqual(idx, v6a.mass_to_bin(15.3))

    def test_mass_token_positive(self):
        tok = self.config.mass_token(15.3)
        self.assertIn("MASS_", tok)

    def test_mass_token_negative(self):
        tok = self.config.mass_token(-14.7)
        self.assertIn("MASS_", tok)

    def test_mass_token_roundtrip(self):
        for delta in [15.3, -14.7, 230.0, -333.0, -888.0]:
            idx = self.config.mass_to_bin(delta)
            tok = self.config.mass_token(delta)
            num_part = tok.replace("MASS_", "")
            self.assertEqual(int(num_part), idx)


# ═════════════════════════════════════════════════════════════════════════════
# Test 2: V6b Data Preparation
# ═════════════════════════════════════════════════════════════════════════════
class TestV6bDataPrep(unittest.TestCase):
    """Test V6b data preparation and OpenNMT format output."""

    @classmethod
    def setUpClass(cls):
        cls.config = V6bConfig()

    def test_kekulize_aromatic(self):
        """Benzene should be kekulized to alternating single/double bonds."""
        result = _kekulize("c1ccccc1")
        self.assertIn("=", result)

    def test_kekulize_non_aromatic(self):
        """Ethanol has no aromatic bonds, should remain similar."""
        result = _kekulize("CCO")
        self.assertIn("C", result)
        self.assertIn("O", result)

    def test_space_chars_basic(self):
        """Char-level spacing of simple SMILES."""
        result = _space_chars("CCO")
        tokens = result.split()
        self.assertGreater(len(tokens), 0)
        self.assertIn("C", tokens)
        self.assertIn("O", tokens)

    def test_space_chars_bracketed(self):
        """Bracketed atoms appear in spaced output."""
        result = _space_chars("[Na+]")
        self.assertIn("[", result)
        self.assertIn("]", result)

    def test_format_input_structure(self):
        """Input format: parent_chars | M A S S _ X"""
        formatted = format_input("CCO", 2.0157, self.config)
        self.assertIn("|", formatted)
        self.assertIn("M", formatted)

    def test_data_prep_outputs_files(self):
        """prepare_v6b_data should output text files."""
        data_dir = REPO_ROOT / "data" / "processed" / "v6b"
        self.assertTrue((data_dir / "src-train.txt").exists(), "src-train.txt should exist")
        self.assertTrue((data_dir / "tgt-train.txt").exists(), "tgt-train.txt should exist")
        self.assertTrue((data_dir / "src-val.txt").exists(), "src-val.txt should exist")
        self.assertTrue((data_dir / "tgt-val.txt").exists(), "tgt-val.txt should exist")

        with open(data_dir / "src-train.txt", encoding="utf-8") as f:
            first_line = f.readline().strip()
            self.assertIn("|", first_line, "Source line should contain | separator")

    def test_src_tgt_line_count_match(self):
        """Source and target files should have same number of lines."""
        for split in ["train", "val", "test"]:
            src = REPO_ROOT / "data" / "processed" / "v6b" / f"src-{split}.txt"
            tgt = REPO_ROOT / "data" / "processed" / "v6b" / f"tgt-{split}.txt"
            if src.exists() and tgt.exists():
                with open(src, encoding="utf-8") as f:
                    n_src = sum(1 for _ in f)
                with open(tgt, encoding="utf-8") as f:
                    n_tgt = sum(1 for _ in f)
                self.assertEqual(n_src, n_tgt, f"Line count mismatch for {split}")


# ═════════════════════════════════════════════════════════════════════════════
# Test 3: V6b Vocab Building
# ═════════════════════════════════════════════════════════════════════════════
class TestV6bVocab(unittest.TestCase):
    """Test vocabulary building."""

    def test_vocab_files(self):
        """After build_vocab, vocab files should exist with special tokens."""
        data_dir = REPO_ROOT / "data" / "processed" / "v6b"

        if not (data_dir / "src_vocab.txt").exists():
            from src.model.v6b.build_vocab import build_vocab
            build_vocab()

        for name in ["src_vocab.txt", "tgt_vocab.txt"]:
            vp = data_dir / name
            self.assertTrue(vp.exists(), f"{name} should exist")
            content = vp.read_text(encoding="utf-8")
            lines = content.strip().split("\n")
            self.assertIn("<unk>", lines, "Vocab should contain <unk>")
            self.assertIn("<s>", lines, "Vocab should contain <s>")
            self.assertIn("</s>", lines, "Vocab should contain </s>")


# ═════════════════════════════════════════════════════════════════════════════
# Test 4: V6b Inference Interface
# ═════════════════════════════════════════════════════════════════════════════
class TestV6bInference(unittest.TestCase):
    """Test V6b inference functions work without trained model."""

    @classmethod
    def setUpClass(cls):
        cls.config = V6bConfig()

    def test_format_input_simple(self):
        result = format_input("CCO", 2.0157, self.config)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        self.assertIn("|", result)

    def test_format_input_roundtrip_info(self):
        """Mass token should encode the correct bin for the given delta."""
        for delta in [15.3, -14.7, 0.0]:
            formatted = format_input("CCO", delta, self.config)
            parts = formatted.split("|")
            self.assertEqual(len(parts), 2)
            mass_part = parts[1].strip().replace(" ", "")
            expected_tok = self.config.mass_token(delta)
            self.assertEqual(expected_tok, mass_part,
                             f"Mass token mismatch for delta={delta}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
