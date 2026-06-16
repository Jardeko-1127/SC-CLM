"""V5A1 test benchmark: same 2x2 CFG×TTA matrix as V5B1 (scratch/run_v5_benchmark_matrix.py)."""

from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval.metrics import V4Metrics
from src.model.v5.inference import V5Inference

TS = "20260518"
BEAMS = 10
GUIDANCE = 1.5
BRANCH = "v5a1"
CKPT = "results/checkpoints/v5a1_lora_full_20260510_1127/best"
BENCH = REPO / "results/benchmark/v5"
TEST_CSV = REPO / "data/processed/test.csv"
VAL_BEST_PPM = 4.26

MATRIX = [
    ("on", "on", 10),
    ("on", "off", 0),
    ("off", "on", 10),
    ("off", "off", 0),
]

# Earlier single-run predictions (cfg-on + tta-on)
LEGACY_PRED = {
    ("on", "on"): BENCH / "v5a1_test_preds_lora_full_20260510_1127.csv",
}


def run_id(cfg: str, tta: str) -> str:
    return f"{BRANCH}__cfg-{cfg}__tta-{tta}__seed-42__beams-{BEAMS}__ts-{TS}"


def pred_path(cfg: str, tta: str) -> Path:
    return BENCH / f"{BRANCH}_test_preds_cfg-{cfg}_tta-{tta}_{TS}.csv"


def evaluate(df: pd.DataFrame, cfg: str, tta: str, tta_n: int) -> dict:
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
    rid = run_id(cfg, tta)
    run_dir = BENCH / "runs" / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": rid,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "branch": BRANCH,
        "checkpoint": CKPT,
        "dataset": "data/processed/test.csv",
        "n_samples": int(len(df)),
        "predictions_csv": str(pred_path(cfg, tta).relative_to(REPO)),
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


def row_from_meta(m: dict) -> dict:
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


def write_summaries(rows: list[dict]) -> None:
    df = pd.DataFrame(rows).sort_values(["cfg", "tta"], ascending=[False, False])
    summary_dir = BENCH / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(summary_dir / "summary_v5a1_test.csv", index=False)

    # Side-by-side with V5B1 (same run_id suffix)
    b1 = pd.read_csv(summary_dir / "summary_main.csv")
    b1 = b1[b1["branch"] == "v5b1"].copy()
    b1 = b1.rename(columns={c: f"v5b1_{c}" if c not in ("cfg", "tta") else c for c in b1.columns})
    a1 = df.rename(columns={c: f"v5a1_{c}" if c not in ("cfg", "tta", "run_id") else c for c in df.columns})
    merged = a1.merge(
        b1,
        on=["cfg", "tta"],
        how="outer",
        suffixes=("", "_dup"),
    )
    merged.to_csv(summary_dir / "summary_v5a1_v5b1_matrix.csv", index=False)

    best_a1 = df.loc[df["ppm_pass_pct"].idxmax()]
    b1df = pd.read_csv(summary_dir / "summary_main.csv")
    b1df = b1df[b1df["branch"] == "v5b1"]
    best_b1 = b1df.loc[b1df["ppm_pass_pct"].idxmax()]
    pd.DataFrame(
        [
            {
                "branch": "v5a1",
                "checkpoint": CKPT,
                "val_best_ppm_pct": VAL_BEST_PPM,
                "test_cfg": "on" if best_a1["cfg"] else "off",
                "test_tta": "on" if best_a1["tta"] else "off",
                "test_beams": BEAMS,
                "test_n": int(best_a1["n_samples"]),
                "validity_pct": best_a1["validity_pct"],
                "ppm_pass_pct": best_a1["ppm_pass_pct"],
                "exact_match_pct": best_a1["exact_match_pct"],
                "skeleton_pct": best_a1["skeleton_pct"],
                "tanimoto_median": best_a1["tanimoto_median"],
                "predictions_csv": pred_path(
                    "on" if best_a1["cfg"] else "off",
                    "on" if best_a1["tta"] else "off",
                ).as_posix(),
            },
            {
                "branch": "v5b1",
                "checkpoint": best_b1["checkpoint"],
                "val_best_ppm_pct": 56.39,
                "test_cfg": "on" if best_b1["cfg"] else "off",
                "test_tta": "on" if best_b1["tta"] else "off",
                "test_beams": BEAMS,
                "test_n": int(best_b1["n_samples"]),
                "validity_pct": best_b1["validity_pct"],
                "ppm_pass_pct": best_b1["ppm_pass_pct"],
                "exact_match_pct": best_b1["exact_match_pct"],
                "skeleton_pct": best_b1["skeleton_pct"],
                "tanimoto_median": best_b1["tanimoto_median"],
                "predictions_csv": f"v5b1_test_preds_cfg-{'on' if best_b1['cfg'] else 'off'}_tta-{'on' if best_b1['tta'] else 'off'}_{TS}.csv",
            },
        ]
    ).to_csv(summary_dir / "summary_v5a1_v5b1_test.csv", index=False)

    # Append v5a1 rows to summary_main (do not drop v5a / v5b1)
    main = pd.read_csv(summary_dir / "summary_main.csv")
    main = main[main["branch"] != "v5a1"]
    main = pd.concat([main, df], ignore_index=True)
    main.sort_values(["branch", "cfg", "tta"], ascending=[True, False, False]).to_csv(
        summary_dir / "summary_main.csv", index=False
    )
    print(f"Wrote {summary_dir}/summary_v5a1_test.csv", flush=True)
    print(f"Wrote {summary_dir}/summary_v5a1_v5b1_matrix.csv", flush=True)
    print(f"Wrote {summary_dir}/summary_v5a1_v5b1_test.csv", flush=True)
    print(f"Updated {summary_dir}/summary_main.csv", flush=True)


def main() -> None:
    df_test = pd.read_csv(TEST_CSV)
    rows: list[dict] = []
    ckpt = str(REPO / CKPT)

    print(f"=== {BRANCH} load once: {ckpt} ===", flush=True)
    t0 = time.time()
    infer = V5Inference(checkpoint_path=ckpt, guidance_scale=GUIDANCE)
    print(f"Loaded in {time.time() - t0:.1f}s", flush=True)

    for cfg, tta_flag, tta_n in MATRIX:
        rid = run_id(cfg, tta_flag)
        out = pred_path(cfg, tta_flag)
        run_metrics = BENCH / "runs" / rid / "metrics.json"

        if run_metrics.is_file():
            print(f"SKIP {rid}", flush=True)
            if not out.is_file():
                legacy = LEGACY_PRED.get((cfg, tta_flag))
                if legacy and legacy.is_file():
                    shutil.copy(legacy, out)
                elif (BENCH / "runs" / rid / "predictions.csv").is_file():
                    shutil.copy(BENCH / "runs" / rid / "predictions.csv", out)
            rows.append(row_from_meta(json.loads(run_metrics.read_text(encoding="utf-8"))))
            continue

        legacy = LEGACY_PRED.get((cfg, tta_flag))
        if legacy and legacy.is_file() and not out.is_file():
            shutil.copy(legacy, out)
        if legacy and legacy.is_file() and not run_metrics.is_file():
            print(f"EVAL legacy -> {rid}", flush=True)
            pred_df = pd.read_csv(out if out.is_file() else legacy)
            pct = evaluate(pred_df, cfg, tta_flag, tta_n)
            rows.append(row_from_meta(json.loads(run_metrics.read_text(encoding="utf-8"))))
            continue

        t1 = time.time()
        print(f"RUN {rid} ...", flush=True)
        result = infer.predict_batch_with_rerank(
            df_test,
            use_cfg=(cfg == "on"),
            tta=tta_n,
            num_beams=BEAMS,
        )
        result.to_csv(out, index=False)
        pct = evaluate(result, cfg, tta_flag, tta_n)
        print(f"  done in {time.time() - t1:.1f}s -> {pct}", flush=True)
        rows.append(row_from_meta(json.loads(run_metrics.read_text(encoding="utf-8"))))

    write_summaries(rows)


if __name__ == "__main__":
    main()
