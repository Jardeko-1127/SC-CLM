"""
V6a data preparation — merge raw Norman + CTS data, compute delta_mz, scaffold-split.

Outputs: data/processed/v6_train.csv, v6_val.csv, v6_test.csv
Columns: parent_smiles, product_smiles, delta_mz, source
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Set

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("v6_prepare_data")

EXCLUDE_RAW = {"S81_THSTPS_Transformations.csv"}
CTS_DIR = Path("results/cts_augmented")
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
REPO_ROOT = Path(__file__).resolve().parents[3]


def _canon(smi: str) -> Optional[str]:
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def _get_inchikey14(smi: str) -> Optional[str]:
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        return AllChem.InchiToInchiKey(AllChem.MolToInchi(m))[:14]
    except Exception:
        return None


def _get_mass(smi: str) -> Optional[float]:
    try:
        m = Chem.MolFromSmiles(smi)
        return Descriptors.ExactMolWt(m) if m else None
    except Exception:
        return None


def _get_scaffold(smi: str) -> str:
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return "__generic__"
        scaf = MurckoScaffold.GetScaffoldForMol(m)
        if scaf is None or scaf.GetNumAtoms() == 0:
            return "__generic__"
        return Chem.MolToSmiles(scaf, canonical=True)
    except Exception:
        return "__generic__"


def load_raw_data() -> pd.DataFrame:
    """Load all raw CSV files except S81_THSTPS."""
    raw_dir = REPO_ROOT / RAW_DIR
    files = sorted(f for f in raw_dir.glob("*.csv") if f.name not in EXCLUDE_RAW)
    logger.info("Loading %d raw CSV files (excluding S81_THSTPS)...", len(files))

    frames = []
    for fp in files:
        try:
            df = pd.read_csv(fp, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(fp, encoding="latin-1")

        # Normalize column names
        for col in ["Predecessor_SMILES", "Successor_SMILES"]:
            if col not in df.columns:
                logger.warning("  %s: missing column %s, skipping", fp.name, col)
                df = None
                break
        if df is None:
            continue

        df = df[["Predecessor_SMILES", "Successor_SMILES"]].copy()
        df.columns = ["parent_smiles", "product_smiles"]
        df["source_file"] = fp.name
        df["source"] = "raw"
        frames.append(df)
        logger.info("  %s: %d rows", fp.name, len(df))

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Raw total: %d rows from %d files", len(combined), len(frames))
    return combined


def load_cts_data() -> pd.DataFrame:
    """Load CTS LIKELY gen=1 combined results."""
    cts_dir = REPO_ROOT / CTS_DIR
    csvs = sorted(cts_dir.glob("cts_all_likely_*.csv"))
    if not csvs:
        logger.warning("No CTS combined output found in %s", cts_dir)
        return pd.DataFrame(columns=["parent_smiles", "product_smiles", "delta_mz", "source"])

    latest = csvs[-1]
    logger.info("Loading CTS data from %s", latest.name)
    df = pd.read_csv(latest)

    # CTS columns: parent_SMILES, SMILES (product), cts_pathway
    df = df[["parent_SMILES", "SMILES", "cts_pathway"]].copy()
    df.columns = ["parent_smiles", "product_smiles", "cts_pathway"]
    df["source"] = "cts_" + df["cts_pathway"].astype(str).str.replace("_ranked", "")
    df.drop(columns=["cts_pathway"], inplace=True)
    logger.info("CTS total: %d rows", len(df))
    return df


def clean_and_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize, filter invalid, compute delta_mz, deduplicate."""
    logger.info("Cleaning and deduplicating %d rows...", len(df))

    df["parent_smiles"] = df["parent_smiles"].apply(_canon)
    df["product_smiles"] = df["product_smiles"].apply(_canon)
    df = df.dropna(subset=["parent_smiles", "product_smiles"])

    # Mass filter
    df["parent_mass"] = df["parent_smiles"].apply(_get_mass)
    df["product_mass"] = df["product_smiles"].apply(_get_mass)
    df = df.dropna(subset=["parent_mass", "product_mass"])
    df = df[(df["parent_mass"] >= 50) & (df["parent_mass"] <= 1200)]
    df = df[(df["product_mass"] >= 50) & (df["product_mass"] <= 1200)]

    df["delta_mz"] = df["product_mass"] - df["parent_mass"]

    # InChIKey-14 dedup
    df["parent_ik14"] = df["parent_smiles"].apply(_get_inchikey14)
    df["product_ik14"] = df["product_smiles"].apply(_get_inchikey14)
    df = df.dropna(subset=["parent_ik14", "product_ik14"])
    before = len(df)
    df = df.drop_duplicates(subset=["parent_ik14", "product_ik14"])
    logger.info("  Dedup: %d → %d (-%d)", before, len(df), before - len(df))

    return df


def scaffold_split(
    df: pd.DataFrame, train_ratio=0.85, val_ratio=0.10, test_ratio=0.05, seed=42
) -> pd.DataFrame:
    """Murcko scaffold-based stratified split."""
    rng = np.random.RandomState(seed)
    df = df.copy()
    df["scaffold"] = df["parent_smiles"].apply(_get_scaffold)

    scaff_groups = df.groupby("scaffold")
    assignments = {}

    for scaf, idxs in scaff_groups.groups.items():
        n = len(idxs)
        if n >= 10:
            # Large: split rows
            idx_list = list(idxs)
            rng.shuffle(idx_list)
            n_val = max(1, int(n * val_ratio))
            for i in idx_list[:n_val]:
                assignments[i] = "val"
            for i in idx_list[n_val:]:
                assignments[i] = "train"
        elif n >= 2:
            # Medium: whole scaffold to one split
            r = rng.random()
            if r < train_ratio:
                split = "train"
            elif r < train_ratio + val_ratio:
                split = "val"
            else:
                split = "test"
            for i in idxs:
                assignments[i] = split
        else:
            # Singletons
            r = rng.random()
            split = "val" if r < 0.5 else "train"
            for i in idxs:
                assignments[i] = split

    df["split"] = df.index.map(assignments)
    df["split"] = df["split"].fillna("train")

    for s in ["train", "val", "test"]:
        n = (df["split"] == s).sum()
        logger.info("  %s: %d rows (%.1f%%)", s, n, n / len(df) * 100)

    return df


def audit_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Check and fix train/val/test parent InChIKey overlap."""
    splits = {}
    for s in ["train", "val", "test"]:
        splits[s] = set(df[df["split"] == s]["parent_ik14"])

    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = splits[a] & splits[b]
        if overlap:
            logger.warning("  Leak %s ↔ %s: %d shared parents", a, b, len(overlap))
            # Move overlapping from smaller split to larger
            if len(splits[a]) >= len(splits[b]):
                splits[b] -= overlap
                mask = (df["split"] == b) & (df["parent_ik14"].isin(overlap))
                df.loc[mask, "split"] = a
            else:
                splits[a] -= overlap
                mask = (df["split"] == a) & (df["parent_ik14"].isin(overlap))
                df.loc[mask, "split"] = b
    return df


def prepare_v6_data():
    """Main entry: merge raw + CTS, clean, split, save."""
    logger.info("====== V6a Data Preparation ======")

    raw_df = load_raw_data()
    cts_df = load_cts_data()

    combined = pd.concat([raw_df, cts_df], ignore_index=True)
    logger.info("Combined: %d rows (raw + CTS)", len(combined))

    cleaned = clean_and_dedup(combined)
    logger.info("After clean: %d rows", len(cleaned))

    df = scaffold_split(cleaned)
    df = audit_leakage(df)

    out_dir = REPO_ROOT / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = ["parent_smiles", "product_smiles", "delta_mz", "source"]
    for split_name in ["train", "val", "test"]:
        sub = df[df["split"] == split_name][cols].copy()
        fp = out_dir / f"v6_{split_name}.csv"
        sub.to_csv(fp, index=False)
        logger.info("Saved %s: %d rows → %s", split_name, len(sub), fp)

    logger.info("====== V6a data preparation complete ======")
    return df


if __name__ == "__main__":
    prepare_v6_data()
