"""Evaluate test predictions with src/eval/metrics.py and write benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.eval.metrics import V4Metrics


def main() -> None:
    p = argparse.ArgumentParser(description="SC-CLM test-set metrics (V4Metrics)")
    p.add_argument("--predictions", required=True, help="CSV with parent/target/prediction/delta_mz")
    p.add_argument("--run-id", required=True)
    p.add_argument("--branch", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cfg", choices=("on", "off"), default="on")
    p.add_argument("--tta", choices=("on", "off"), default="on")
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--beams", type=int, default=10)
    p.add_argument("--tta-n", type=int, default=10)
    args = p.parse_args()

    df = pd.read_csv(args.predictions)
    if "target_smiles" not in df.columns and "product_smiles" in df.columns:
        df["target_smiles"] = df["product_smiles"]

    metrics = V4Metrics().evaluate_batch(df)
    pct = {
        "validity_rate": round(metrics["validity_rate"] * 100, 2),
        "ppm_pass_rate": round(metrics["ppm_pass_rate"] * 100, 2),
        "skeleton_pass_rate": round(metrics["skeleton_pass_rate"] * 100, 2),
        "exact_match_rate": round(metrics["exact_match_rate"] * 100, 2),
        "tanimoto_median": round(metrics["tanimoto_median"], 4),
    }

    run_dir = Path("results/benchmark/v5/runs") / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "run_id": args.run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "branch": args.branch,
        "checkpoint": args.checkpoint,
        "dataset": "data/processed/test.csv",
        "n_samples": int(len(df)),
        "predictions_csv": str(args.predictions),
        "cfg_enabled": args.cfg == "on",
        "cfg_scale": args.guidance if args.cfg == "on" else None,
        "tta_enabled": args.tta == "on",
        "tta_n": args.tta_n if args.tta == "on" else 1,
        "decode_params": {"num_beams": args.beams},
        "metrics_script": "src/eval/metrics.py",
        "metrics": pct,
    }

    (run_dir / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    df.to_csv(run_dir / "predictions.csv", index=False)

    print(json.dumps(pct, indent=2))
    print(f"Wrote {run_dir}")


if __name__ == "__main__":
    main()
