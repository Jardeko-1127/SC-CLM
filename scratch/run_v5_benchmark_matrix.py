"""Run V5 mainline 2x2 benchmark (v5a/v5b x cfg on/off x tta on/off) on test.csv."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval.metrics import V4Metrics
from src.model.v5.inference import V5Inference

TS = "20260519"
BEAMS = 10
GUIDANCE = 1.5
TEST_CSV = REPO / "data/processed/test.csv"
BENCH = REPO / "results/benchmark/v5"

# Mainline bundles from sc_clm_v5a_v5b_best_20260513_170450.tar.gz
MODELS = {
    "v5a": {
        "checkpoint": "results/checkpoints/v5a/best",
        "tokenizer_source": "results/checkpoints/v5a/best",
    },
    "v5b": {
        "checkpoint": "results/checkpoints/v5b/best",
        "tokenizer_source": None,
    },
}

KEEP_IN_SUMMARY = frozenset({"v5a1", "v5b1"})

MATRIX = [
    ("on", "on", 10),
    ("on", "off", 0),
    ("off", "on", 10),
    ("off", "off", 0),
]


def run_id(branch: str, cfg: str, tta: str) -> str:
    return f"{branch}__cfg-{cfg}__tta-{tta}__seed-42__beams-{BEAMS}__ts-{TS}"


def pred_path(branch: str, cfg: str, tta: str) -> Path:
    return BENCH / f"{branch}_test_preds_cfg-{cfg}_tta-{tta}_{TS}.csv"


def evaluate(df: pd.DataFrame, branch: str, cfg: str, tta: str, tta_n: int, ckpt: str) -> dict:
    if "target_smiles" not in df.columns and "product_smiles" in df.columns:
        df = df.copy()
        df["target_smiles"] = df["product_smiles"]
    metrics = V4Metrics().evaluate_batch(df)
    pct = {
        "validity_rate": round(metrics["validity_rate"] * 100, 2),
        "ppm_pass_rate": round(metrics["ppm_pass_rate"] * 100, 2),
        "skeleton_pass_rate": round(metrics["skeleton_pass_rate"] * 100, 2),
        "exact_match_rate": round(metrics["exact_match_rate"] * 100, 2),
        "tanimoto_median": round(metrics["tanimoto_median"], 4),
    }
    rid = run_id(branch, cfg, tta)
    run_dir = BENCH / "runs" / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": rid,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "branch": branch,
        "checkpoint": ckpt,
        "dataset": "data/processed/test.csv",
        "n_samples": int(len(df)),
        "predictions_csv": str(pred_path(branch, cfg, tta).relative_to(REPO)),
        "cfg_enabled": cfg == "on",
        "cfg_scale": GUIDANCE if cfg == "on" else None,
        "tta_enabled": tta == "on",
        "tta_n": tta_n if tta == "on" else 1,
        "decode_params": {"num_beams": BEAMS},
        "metrics_script": "src/eval/metrics.py",
        "metrics": pct,
    }
    (run_dir / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    df.to_csv(run_dir / "predictions.csv", index=False)
    return pct


def main() -> None:
    df = pd.read_csv(TEST_CSV)
    summary_rows = []

    for branch, spec in MODELS.items():
        ckpt = str(REPO / spec["checkpoint"])
        tok = spec["tokenizer_source"]
        tok_path = str(REPO / tok) if tok else None
        print(f"\n=== {branch} load once: {ckpt} ===", flush=True)
        t0 = time.time()
        infer = V5Inference(
            checkpoint_path=ckpt,
            guidance_scale=GUIDANCE,
            tokenizer_source=tok_path,
        )
        print(f"Loaded in {time.time() - t0:.1f}s", flush=True)

        for cfg, tta_flag, tta_n in MATRIX:
            rid = run_id(branch, cfg, tta_flag)
            out = pred_path(branch, cfg, tta_flag)
            if (BENCH / "runs" / rid / "metrics.json").is_file() and out.is_file():
                print(f"SKIP {rid} (exists)", flush=True)
                m = json.loads((BENCH / "runs" / rid / "metrics.json").read_text(encoding="utf-8"))
                summary_rows.append(_row_from_meta(m))
                continue

            t1 = time.time()
            print(f"RUN {rid} ...", flush=True)
            result = infer.predict_batch_with_rerank(
                df,
                use_cfg=(cfg == "on"),
                tta=tta_n,
                num_beams=BEAMS,
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(out, index=False)
            pct = evaluate(result, branch, cfg, tta_flag, tta_n, spec["checkpoint"])
            print(f"  done in {time.time() - t1:.1f}s -> {pct}", flush=True)
            summary_rows.append(
                {
                    "run_id": rid,
                    "branch": branch,
                    "checkpoint": spec["checkpoint"],
                    "n_samples": len(df),
                    "cfg": cfg == "on",
                    "tta": tta_flag == "on",
                    "tta_n": tta_n if tta_flag == "on" else 1,
                    "guidance": GUIDANCE if cfg == "on" else None,
                    "beams": BEAMS,
                    "validity_pct": pct["validity_rate"],
                    "ppm_pass_pct": pct["ppm_pass_rate"],
                    "exact_match_pct": pct["exact_match_rate"],
                    "skeleton_pct": pct["skeleton_pass_rate"],
                    "tanimoto_median": pct["tanimoto_median"],
                }
            )

    out_csv = BENCH / "summary" / "summary_main.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged = _merge_summary(summary_rows, out_csv)
    pd.DataFrame(merged).drop_duplicates(subset=["run_id"]).sort_values(["branch", "cfg", "tta"]).to_csv(
        out_csv, index=False
    )
    print(f"\nWrote {out_csv} ({len(merged)} rows)", flush=True)


def _row_from_meta(m: dict) -> dict:
    met = m["metrics"]
    return {
        "run_id": m["run_id"],
        "branch": m["branch"],
        "checkpoint": m["checkpoint"],
        "n_samples": m["n_samples"],
        "cfg": m["cfg_enabled"],
        "tta": m["tta_enabled"],
        "tta_n": m["tta_n"],
        "guidance": m.get("cfg_scale"),
        "beams": m["decode_params"]["num_beams"],
        "validity_pct": met["validity_rate"],
        "ppm_pass_pct": met["ppm_pass_rate"],
        "exact_match_pct": met["exact_match_rate"],
        "skeleton_pct": met["skeleton_pass_rate"],
        "tanimoto_median": met["tanimoto_median"],
    }


def _merge_summary(new_rows: list, out_csv: Path) -> list:
    """Keep LoRA rows from prior summary_main; replace v5a/v5b with this run."""
    kept: list[dict] = []
    if out_csv.is_file():
        old = pd.read_csv(out_csv)
        for _, r in old.iterrows():
            if r.get("branch") in KEEP_IN_SUMMARY:
                kept.append(r.to_dict())
    have = {r["run_id"] for r in kept}
    for row in new_rows:
        if row["run_id"] not in have:
            kept.append(row)
    return kept


if __name__ == "__main__":
    main()
