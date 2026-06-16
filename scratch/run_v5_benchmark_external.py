"""Phase E: BioTransformer + EPA CTS on test.csv -> summary_external.csv."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval.benchmark_bt import BioTransformerBenchmark, BT_JAR
from src.eval.benchmark_cts import CTSBenchmark
from src.eval.metrics import V4Metrics

TEST_CSV = REPO / "data/processed/test.csv"
BENCH = REPO / "results/benchmark/v5"
TS = "20260519"


def evaluate_preds(df: pd.DataFrame, tool: str, runtime_sec: float) -> dict:
    if "target_smiles" not in df.columns and "product_smiles" in df.columns:
        df = df.copy()
        df["target_smiles"] = df["product_smiles"]
    m = V4Metrics().evaluate_batch(df)
    return {
        "model_or_tool": tool,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": "data/processed/test.csv",
        "n_samples": len(df),
        "top_k": 1,
        "setting": "native_input_ppm_rerank",
        "validity_pct": round(m["validity_rate"] * 100, 2),
        "exact_match_pct": round(m["exact_match_rate"] * 100, 2),
        "ppm_pass_pct": round(m["ppm_pass_rate"] * 100, 2),
        "tanimoto_median": round(m["tanimoto_median"], 4),
        "eval_runtime_sec": round(runtime_sec, 2),
        "samples_per_sec": round(len(df) / runtime_sec, 4) if runtime_sec > 0 else None,
        "latency_ms_per_sample": round(runtime_sec / len(df) * 1000, 2) if len(df) else None,
    }


def run_biotransformer() -> dict | None:
    jar = Path(BT_JAR)
    if not jar.is_file():
        print(f"SKIP BioTransformer: missing {jar}", flush=True)
        return None
    out = BENCH / f"biotransformer_test_preds_{TS}.csv"
    bench = BioTransformerBenchmark()
    t0 = time.perf_counter()
    bench.run_benchmark(str(TEST_CSV), str(out))
    elapsed = time.perf_counter() - t0
    df = pd.read_csv(out)
    if "delta_mz" not in df.columns:
        test = pd.read_csv(TEST_CSV)
        df["delta_mz"] = test["delta_mz"].values
    row = evaluate_preds(df, "biotransformer", elapsed)
    row["predictions_csv"] = str(out.relative_to(REPO))
    print(f"BioTransformer: {row}", flush=True)
    return row


def run_cts() -> dict | None:
    out = BENCH / f"epa_cts_test_preds_{TS}.csv"
    bench = CTSBenchmark()
    t0 = time.perf_counter()
    try:
        bench.run_benchmark(str(TEST_CSV), str(out))
    except Exception as e:
        print(f"CTS benchmark failed: {e}", flush=True)
        return None
    elapsed = time.perf_counter() - t0
    if not out.is_file():
        return None
    df = pd.read_csv(out)
    if "delta_mz" not in df.columns:
        test = pd.read_csv(TEST_CSV)
        df["delta_mz"] = test["delta_mz"].values
    row = evaluate_preds(df, "epa_cts", elapsed)
    row["predictions_csv"] = str(out.relative_to(REPO))
    print(f"EPA CTS: {row}", flush=True)
    return row


def add_sc_clm_reference(rows: list[dict]) -> None:
    main = BENCH / "summary" / "summary_main.csv"
    if not main.is_file():
        return
    df = pd.read_csv(main)
    for branch in ("v5b", "v5b1"):
        sub = df[(df["branch"] == branch) & (df["cfg"] == True) & (df["tta"] == True)]
        if sub.empty:
            continue
        r = sub.iloc[0]
        seeds = BENCH / "summary" / "summary_seeds.csv"
        runtime = None
        if seeds.is_file():
            sd = pd.read_csv(seeds)
            ssub = sd[sd["branch"] == branch]
            if not ssub.empty:
                runtime = float(ssub["eval_runtime_sec"].mean())
        rows.append(
            {
                "model_or_tool": branch,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "dataset": "data/processed/test.csv",
                "n_samples": int(r["n_samples"]),
                "top_k": 1,
                "setting": "cfg-on_tta-on",
                "validity_pct": float(r["validity_pct"]),
                "exact_match_pct": float(r["exact_match_pct"]),
                "ppm_pass_pct": float(r["ppm_pass_pct"]),
                "tanimoto_median": float(r["tanimoto_median"]),
                "eval_runtime_sec": runtime,
                "samples_per_sec": round(int(r["n_samples"]) / runtime, 4) if runtime else None,
                "latency_ms_per_sample": round(runtime / int(r["n_samples"]) * 1000, 2) if runtime else None,
                "predictions_csv": f"(see summary_main {branch})",
            }
        )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["bt", "cts", "all"], default="all")
    args = ap.parse_args()

    rows: list[dict] = []
    if args.only in ("bt", "all"):
        r = run_biotransformer()
        if r:
            rows.append(r)
    if args.only in ("cts", "all"):
        r = run_cts()
        if r:
            rows.append(r)
    add_sc_clm_reference(rows)

    out = BENCH / "summary" / "summary_external.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
