"""Phase B: multi-seed stability + reliability subsets + efficiency (top models)."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval.metrics import V4Metrics
from src.model.v5.inference import V5Inference

TS = "20260519"
BEAMS = 10
GUIDANCE = 1.5
SEEDS = [42, 123, 456]
TEST_CSV = REPO / "data/processed/test.csv"
TRAIN_CSV = REPO / "data/processed/train.csv"
BENCH = REPO / "results/benchmark/v5"

TOP_MODELS = {
    "v5b": {
        "checkpoint": "results/checkpoints/v5b/best",
        "tokenizer_source": None,
    },
    "v5b1": {
        "checkpoint": "results/checkpoints/v5b1_lora_full_20260513_1029/best",
        "tokenizer_source": None,
    },
}

# Best inference config from phase A
CFG, TTA_FLAG, TTA_N = "on", "on", 10


def murcko(smi: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return None


def target_mass(parent: str, delta: float) -> float | None:
    try:
        mol = Chem.MolFromSmiles(parent)
        if mol is None:
            return None
        return Descriptors.ExactMolWt(mol) + delta
    except Exception:
        return None


def build_subsets(test_df: pd.DataFrame, train_df: pd.DataFrame) -> dict[str, pd.Series]:
    train_sc = {murcko(s) for s in train_df["parent_smiles"] if murcko(s)}
    scaffolds = test_df["parent_smiles"].map(murcko)
    ood = scaffolds.map(lambda s: s is not None and s not in train_sc)

    plen = test_df["parent_smiles"].str.len()
    long_thr = float(plen.quantile(0.75))
    long_smiles = plen >= long_thr

    tm = test_df.apply(lambda r: target_mass(r["parent_smiles"], float(r["delta_mz"])), axis=1)
    tm_s = pd.to_numeric(tm, errors="coerce")
    q25, q75 = tm_s.quantile(0.25), tm_s.quantile(0.75)
    ppm_boundary = (tm_s <= q25) | (tm_s >= q75)

    return {
        "full_test": pd.Series(True, index=test_df.index),
        "ood_scaffold": ood,
        "long_smiles": long_smiles,
        "ppm_boundary_mass": ppm_boundary.fillna(False),
    }


def failure_breakdown(df: pd.DataFrame) -> dict[str, int]:
    m = V4Metrics()
    invalid = mass_mismatch = structure_mismatch = 0
    for _, row in df.iterrows():
        parent, target, pred, delta = (
            row["parent_smiles"],
            row["target_smiles"],
            row["prediction"],
            float(row["delta_mz"]),
        )
        if not m.check_validity(pred):
            invalid += 1
            continue
        if not m.check_ppm_fidelity(parent, pred, delta):
            mass_mismatch += 1
            continue
        tik, pik = m.get_inchikey_14(target), m.get_inchikey_14(pred)
        if tik and pik and tik == pik:
            continue
        structure_mismatch += 1
    return {
        "invalid": invalid,
        "mass_mismatch": mass_mismatch,
        "structure_mismatch": structure_mismatch,
    }


def evaluate_subset(df: pd.DataFrame, mask: pd.Series) -> dict | None:
    sub = df.loc[mask].copy()
    if len(sub) == 0:
        return None
    if "target_smiles" not in sub.columns and "product_smiles" in sub.columns:
        sub["target_smiles"] = sub["product_smiles"]
    metrics = V4Metrics().evaluate_batch(sub)
    fails = failure_breakdown(sub)
    n = len(sub)
    return {
        "n_samples": n,
        "validity_pct": round(metrics["validity_rate"] * 100, 2),
        "ppm_pass_pct": round(metrics["ppm_pass_rate"] * 100, 2),
        "exact_match_pct": round(metrics["exact_match_rate"] * 100, 2),
        "skeleton_pct": round(metrics["skeleton_pass_rate"] * 100, 2),
        "tanimoto_median": round(metrics["tanimoto_median"], 4),
        **fails,
    }


def run_id(branch: str, seed: int) -> str:
    return f"{branch}__cfg-on__tta-on__seed-{seed}__beams-{BEAMS}__ts-{TS}"


def pred_path(branch: str, seed: int) -> Path:
    return BENCH / f"{branch}_test_preds_cfg-on_tta-on_seed-{seed}_{TS}.csv"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_seeds() -> list[dict]:
    test_df = pd.read_csv(TEST_CSV)
    rows: list[dict] = []

    for branch, spec in TOP_MODELS.items():
        ckpt = str(REPO / spec["checkpoint"])
        tok = str(REPO / spec["tokenizer_source"]) if spec["tokenizer_source"] else None
        print(f"\n=== {branch} load once: {ckpt} ===", flush=True)
        infer = V5Inference(checkpoint_path=ckpt, guidance_scale=GUIDANCE, tokenizer_source=tok)

        for seed in SEEDS:
            rid = run_id(branch, seed)
            out = pred_path(branch, seed)
            run_dir = BENCH / "runs" / rid
            if run_dir.joinpath("metrics.json").is_file() and out.is_file():
                print(f"SKIP {rid}", flush=True)
                meta = json.loads(run_dir.joinpath("metrics.json").read_text(encoding="utf-8"))
                rows.append(_row_from_meta(meta, seed))
                continue

            set_seed(seed)
            t0 = time.perf_counter()
            print(f"RUN {rid} seed={seed} ...", flush=True)
            result = infer.predict_batch_with_rerank(
                test_df, use_cfg=True, tta=TTA_N, num_beams=BEAMS
            )
            runtime = time.perf_counter() - t0
            out.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(out, index=False)

            if "target_smiles" not in result.columns:
                result["target_smiles"] = test_df["product_smiles"].values
            metrics = V4Metrics().evaluate_batch(result)
            pct = {k: round(v * 100, 2) if "rate" in k else round(v, 4) for k, v in metrics.items()}
            if "validity_rate" in metrics:
                pct = {
                    "validity_rate": round(metrics["validity_rate"] * 100, 2),
                    "ppm_pass_rate": round(metrics["ppm_pass_rate"] * 100, 2),
                    "skeleton_pass_rate": round(metrics["skeleton_pass_rate"] * 100, 2),
                    "exact_match_rate": round(metrics["exact_match_rate"] * 100, 2),
                    "tanimoto_median": round(metrics["tanimoto_median"], 4),
                }
            meta = {
                "run_id": rid,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "branch": branch,
                "checkpoint": spec["checkpoint"],
                "seed": seed,
                "dataset": "data/processed/test.csv",
                "n_samples": len(result),
                "cfg_enabled": True,
                "cfg_scale": GUIDANCE,
                "tta_enabled": True,
                "tta_n": TTA_N,
                "decode_params": {"num_beams": BEAMS},
                "eval_runtime_sec": round(runtime, 2),
                "metrics": pct,
            }
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            result.to_csv(run_dir / "predictions.csv", index=False)
            rows.append(_row_from_meta(meta, seed))
            print(f"  {pct} runtime={runtime:.1f}s", flush=True)

    out = BENCH / "summary" / "summary_seeds.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out} ({len(rows)} rows)", flush=True)
    return rows


def _row_from_meta(meta: dict, seed: int) -> dict:
    m = meta["metrics"]
    return {
        "run_id": meta["run_id"],
        "branch": meta["branch"],
        "checkpoint": meta["checkpoint"],
        "seed": seed,
        "n_samples": meta["n_samples"],
        "cfg": True,
        "tta": True,
        "eval_runtime_sec": meta.get("eval_runtime_sec"),
        "validity_pct": m.get("validity_rate", m.get("validity_pct")),
        "ppm_pass_pct": m.get("ppm_pass_rate", m.get("ppm_pass_pct")),
        "exact_match_pct": m.get("exact_match_rate", m.get("exact_match_pct")),
        "skeleton_pct": m.get("skeleton_pass_rate", m.get("skeleton_pct")),
        "tanimoto_median": m["tanimoto_median"],
    }


def run_reliability() -> list[dict]:
    test_df = pd.read_csv(TEST_CSV)
    train_df = pd.read_csv(TRAIN_CSV)
    subsets = build_subsets(test_df, train_df)

    pred_files = {
        "v5b": BENCH / "v5b_test_preds_cfg-on_tta-on_20260519.csv",
        "v5b1": BENCH / "v5b1_test_preds_cfg-on_tta-on_20260518.csv",
    }
    rows: list[dict] = []
    for branch, p in pred_files.items():
        if not p.is_file():
            print(f"SKIP reliability {branch}: missing {p}", flush=True)
            continue
        pred = pd.read_csv(p)
        if "target_smiles" not in pred.columns:
            pred["target_smiles"] = test_df["product_smiles"].values
        base_rid = f"{branch}__cfg-on__tta-on__seed-42__beams-{BEAMS}__ts-20260518"
        if branch == "v5b":
            base_rid = f"{branch}__cfg-on__tta-on__seed-42__beams-{BEAMS}__ts-20260519"
        for name, mask in subsets.items():
            stats = evaluate_subset(pred, mask)
            if stats is None:
                continue
            rows.append({"run_id": base_rid, "branch": branch, "subset_name": name, **stats})
            print(f"{branch}/{name}: n={stats['n_samples']} ppm={stats['ppm_pass_pct']}%", flush=True)

    out = BENCH / "summary" / "summary_reliability.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out} ({len(rows)} rows)", flush=True)
    return rows


def run_efficiency() -> list[dict]:
    test_df = pd.read_csv(TEST_CSV)
    configs = [
        ("baseline", False, 0),
        ("best", True, TTA_N),
    ]
    rows: list[dict] = []

    for branch, spec in TOP_MODELS.items():
        ckpt = str(REPO / spec["checkpoint"])
        tok = str(REPO / spec["tokenizer_source"]) if spec["tokenizer_source"] else None
        infer = V5Inference(checkpoint_path=ckpt, guidance_scale=GUIDANCE, tokenizer_source=tok)
        set_seed(42)
        basetime = None

        for label, use_cfg, tta in configs:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = infer.predict_batch_with_rerank(
                test_df, use_cfg=use_cfg, tta=tta, num_beams=BEAMS
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            n = len(test_df)
            if label == "baseline":
                basetime = elapsed
            peak_gb = None
            if torch.cuda.is_available():
                peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)
                torch.cuda.reset_peak_memory_stats()
            rows.append(
                {
                    "branch": branch,
                    "config_label": label,
                    "cfg": use_cfg,
                    "tta_n": tta if use_cfg or tta else 1,
                    "n_samples": n,
                    "eval_runtime_sec": round(elapsed, 2),
                    "samples_per_sec": round(n / elapsed, 4),
                    "latency_ms_per_sample": round(elapsed / n * 1000, 2),
                    "peak_vram_gb": peak_gb,
                    "time_multiplier_vs_baseline": round(elapsed / basetime, 3)
                    if basetime and label != "baseline"
                    else 1.0,
                }
            )
            print(
                f"{branch}/{label}: {elapsed:.1f}s ({n/elapsed:.3f} samp/s) vram={peak_gb}GB",
                flush=True,
            )

    out = BENCH / "summary" / "summary_efficiency.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}", flush=True)
    return rows


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only",
        choices=["seeds", "reliability", "efficiency", "all"],
        default="all",
    )
    args = ap.parse_args()
    if args.only in ("reliability", "all"):
        run_reliability()
    if args.only in ("efficiency", "all"):
        run_efficiency()
    if args.only in ("seeds", "all"):
        run_seeds()


if __name__ == "__main__":
    main()
