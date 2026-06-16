"""Evaluate patRoon BioTransformer / CTS / library exports vs test.csv (V4Metrics, PPM rerank).

Reads: results/benchmark/v5/patoon/patoon_*.csv
Writes:
  - predictions_*_{date}.csv
  - summary/summary_external_patoon.csv (all strategies)
  - summary/summary_external_patoon_comparison.csv (per-pathway + V5, merged metrics & coverage)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval.metrics import V4Metrics

TEST_CSV = REPO / "data/processed/test.csv"
PATOON_DIR = REPO / "results/benchmark/v5/patoon"
OUT_DIR = REPO / "results/benchmark/v5"
SUMMARY_MAIN = OUT_DIR / "summary" / "summary_main.csv"
PPM_THRESHOLD = 10.0

BT_PATHWAY_FILES = {
    "ecbased": "patoon_bt_ecbased_*.csv",
    "cyp450": "patoon_bt_cyp450_*.csv",
    "hgut": "patoon_bt_hgut_*.csv",
    "superbio": "patoon_bt_superbio_*.csv",
    "allHuman": "patoon_bt_allhuman_*.csv",
    "env": "patoon_bt_env_*.csv",
}
CTS_PATHWAY_FILES = {
    "abiotic_reduction": "patoon_cts_abiotic_reduction_*.csv",
    "hydrolysis": "patoon_cts_hydrolysis_*.csv",
    "photolysis_unranked": "patoon_cts_photolysis_unranked_*.csv",
    "photolysis_ranked": "patoon_cts_photolysis_ranked_*.csv",
}
LIBRARY_FILE = "patoon_library_default_*.csv"

TOKEN_TO_BT_PATHWAYS: dict[str, list[str]] = {
    "[TRANS_OXIDATION]": ["cyp450", "allHuman"],
    "[TRANS_DI_OXIDATION]": ["cyp450", "allHuman"],
    "[TRANS_DEMETHYLATION]": ["allHuman", "ecbased"],
    "[TRANS_DEETHYLATION]": ["allHuman", "ecbased"],
    "[TRANS_DEHYDROGENATION]": ["allHuman", "cyp450"],
    "[TRANS_GLUCURONIDATION]": ["ecbased", "allHuman"],
    "[TRANS_ACETYLATION]": ["allHuman"],
    "[TRANS_SULFONATION]": ["ecbased", "allHuman"],
    "[TRANS_HYDRATION]": ["allHuman", "env"],
    "[TRANS_ISOMERIZATION]": ["allHuman", "env"],
}

BROAD_BT_PATHWAYS = ["allHuman", "env"]

# 用户要求的逐项对照（单途径 / library / CTS 分库）
COMPARISON_SINGLE_PATHWAYS: list[tuple[str, str, str]] = [
    ("bt_patoon_allHuman_s2", "single:allHuman", "allHuman"),
    ("bt_patoon_superbio_s2", "single:superbio", "superbio"),
    ("bt_patoon_cyp450_s2", "single:cyp450", "cyp450"),
    ("bt_patoon_hgut_s2", "single:hgut", "hgut"),
    ("bt_patoon_env_s2", "single:env", "env"),
    ("bt_patoon_ecbased_s2", "single:ecbased", "ecbased"),
    ("patoon_library_s2", "single:library", "library"),
]
for cts_id in CTS_PATHWAY_FILES:
    COMPARISON_SINGLE_PATHWAYS.append(
        (f"cts_patoon_{cts_id}_s2", f"cts:{cts_id}", cts_id)
    )


def load_test() -> pd.DataFrame:
    df = pd.read_csv(TEST_CSV)
    df = df.reset_index(drop=True)
    df["sample_id"] = [f"test_{i:02d}" for i in range(1, len(df) + 1)]
    df["target_smiles"] = df["product_smiles"]
    return df


def _latest_glob(pattern: str) -> Path | None:
    hits = sorted(PATOON_DIR.glob(pattern))
    return hits[-1] if hits else None


def load_patoon_long() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    all_files = {**BT_PATHWAY_FILES, **CTS_PATHWAY_FILES, "library": LIBRARY_FILE}
    for pathway_id, pattern in all_files.items():
        path = _latest_glob(pattern)
        if path is None:
            print(f"SKIP missing: {pattern}", flush=True)
            continue
        part = pd.read_csv(path)
        part["pathway_id"] = pathway_id
        if pathway_id == "library":
            part["pathway_source"] = "library"
        elif pathway_id in BT_PATHWAY_FILES:
            part["pathway_source"] = "biotransformer"
        else:
            part["pathway_source"] = "cts"
        part["source_file"] = path.name
        if "SMILES" in part.columns:
            part["pred_smiles"] = part["SMILES"].astype(str)
        else:
            part["pred_smiles"] = ""
        rows.append(part)
    if not rows:
        raise FileNotFoundError(f"No patoon CSV under {PATOON_DIR}")
    long = pd.concat(rows, ignore_index=True)
    long["pred_smiles"] = long["pred_smiles"].replace({"nan": "", "None": ""})
    return long


def canon_smiles(smi: str, metrics: V4Metrics) -> str:
    s = str(smi).strip()
    if not s or s.lower() == "nan":
        return ""
    return metrics.canon(s) or s


def ppm_error(parent: str, pred: str, delta: float) -> float | None:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    try:
        p_mol = Chem.MolFromSmiles(parent)
        pr_mol = Chem.MolFromSmiles(pred)
        if not (p_mol and pr_mol):
            return None
        p_mass = Descriptors.ExactMolWt(p_mol)
        pr_mass = Descriptors.ExactMolWt(pr_mol)
        target_mass = p_mass + float(delta)
        if target_mass <= 0:
            return None
        actual_delta = pr_mass - p_mass
        return abs(actual_delta - float(delta)) / target_mass * 1e6
    except Exception:
        return None


def dedupe_candidates(candidates: list[str], metrics: V4Metrics) -> list[str]:
    uniq: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        c = str(c).strip()
        if not c or c.lower() == "nan":
            continue
        key = canon_smiles(c, metrics)
        if key and key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def rank_candidates_ppm(
    parent: str,
    delta: float,
    candidates: list[str],
    metrics: V4Metrics,
) -> list[str]:
    """PPM 升序；合法且可算 PPM 的优先，其余排在后面。"""
    uniq = dedupe_candidates(candidates, metrics)
    if not uniq:
        return []

    scored_valid: list[tuple[float, str]] = []
    scored_other: list[tuple[float, str]] = []
    for c in uniq:
        pe = ppm_error(parent, c, delta)
        if pe is None:
            scored_other.append((1e18, c))
        elif metrics.check_validity(c):
            scored_valid.append((pe, c))
        else:
            scored_other.append((pe, c))

    scored_valid.sort(key=lambda x: x[0])
    scored_other.sort(key=lambda x: x[0])
    return [c for _, c in scored_valid + scored_other]


def pick_best_prediction(
    parent: str,
    delta: float,
    candidates: list[str],
    metrics: V4Metrics,
) -> tuple[str, float | None, int]:
    ranked = rank_candidates_ppm(parent, delta, candidates, metrics)
    if not ranked:
        return "", None, 0

    for c in ranked:
        if metrics.check_validity(c):
            pe = ppm_error(parent, c, delta)
            if pe is not None:
                return c, pe, len(ranked)

    c = ranked[0]
    return c, ppm_error(parent, c, delta), len(ranked)


def candidates_for_sample(
    long: pd.DataFrame,
    sample_id: str,
    pathway_ids: list[str],
) -> list[str]:
    sub = long[(long["parent"] == sample_id) & (long["pathway_id"].isin(pathway_ids))]
    return sub["pred_smiles"].tolist()


def pathway_ids_for_strategy(strategy: str) -> list[str]:
    if strategy == "narrow":
        return sorted({p for v in TOKEN_TO_BT_PATHWAYS.values() for p in v})
    if strategy == "broad":
        return BROAD_BT_PATHWAYS
    if strategy.startswith("single:"):
        return [strategy.split(":", 1)[1]]
    if strategy.startswith("cts:"):
        return [strategy.split(":", 1)[1]]
    if strategy == "cts_union":
        return list(CTS_PATHWAY_FILES.keys())
    return []


def build_predictions(
    test: pd.DataFrame,
    long: pd.DataFrame,
    strategy: str,
) -> pd.DataFrame:
    metrics = V4Metrics()
    out_rows = []

    for _, row in test.iterrows():
        sid = row["sample_id"]
        token = row["token"]
        if strategy == "narrow":
            pids = TOKEN_TO_BT_PATHWAYS.get(token, ["env"])
        else:
            pids = pathway_ids_for_strategy(strategy)

        cands = candidates_for_sample(long, sid, pids)
        pred, pe, n_cand = pick_best_prediction(
            row["parent_smiles"], row["delta_mz"], cands, metrics
        )
        out_rows.append(
            {
                "sample_id": sid,
                "parent_smiles": row["parent_smiles"],
                "token": token,
                "delta_mz": row["delta_mz"],
                "target_smiles": row["target_smiles"],
                "prediction": pred,
                "ppm_error": pe,
                "n_candidates": n_cand,
                "pathways_used": ",".join(pids),
                "strategy": strategy,
            }
        )

    return pd.DataFrame(out_rows)


def evaluate_preds(df: pd.DataFrame) -> dict:
    metrics = V4Metrics().evaluate_batch(df)
    return {
        "validity_pct": round(metrics["validity_rate"] * 100, 2),
        "ppm_pass_pct": round(metrics["ppm_pass_rate"] * 100, 2),
        "exact_match_pct": round(metrics["exact_match_rate"] * 100, 2),
        "skeleton_pct": round(metrics["skeleton_pass_rate"] * 100, 2),
        "tanimoto_median": round(metrics["tanimoto_median"], 4),
    }


def recall_at_k(
    test: pd.DataFrame,
    long: pd.DataFrame,
    pathway_ids: list[str],
    k: int = 5,
    *,
    per_sample_pathways: bool = False,
) -> float:
    metrics = V4Metrics()
    hits = 0
    for _, row in test.iterrows():
        sid = row["sample_id"]
        target = row["target_smiles"]
        target_ik = metrics.get_inchikey_14(target)
        target_canon = metrics.canon(target)
        pids = pathway_ids
        if per_sample_pathways:
            pids = TOKEN_TO_BT_PATHWAYS.get(row["token"], ["env"])
        cands = candidates_for_sample(long, sid, pids)
        ranked = rank_candidates_ppm(
            row["parent_smiles"], row["delta_mz"], cands, metrics
        )[:k]
        ok = False
        for c in ranked:
            if metrics.canon(c) == target_canon and target_canon:
                ok = True
                break
            ik = metrics.get_inchikey_14(c)
            if target_ik and ik and ik == target_ik:
                ok = True
                break
        if ok:
            hits += 1
    return round(hits / len(test) * 100, 2) if len(test) else 0.0


def recall_at_k_ppm(
    test: pd.DataFrame,
    long: pd.DataFrame,
    pathway_ids: list[str],
    k: int = 5,
    *,
    per_sample_pathways: bool = False,
) -> float:
    metrics = V4Metrics()
    hits = 0
    for _, row in test.iterrows():
        sid = row["sample_id"]
        pids = pathway_ids
        if per_sample_pathways:
            pids = TOKEN_TO_BT_PATHWAYS.get(row["token"], ["env"])
        cands = candidates_for_sample(long, sid, pids)
        ranked = rank_candidates_ppm(
            row["parent_smiles"], row["delta_mz"], cands, metrics
        )[:k]
        ok = any(
            metrics.check_validity(c)
            and metrics.check_ppm_fidelity(
                row["parent_smiles"], c, row["delta_mz"], threshold=PPM_THRESHOLD
            )
            for c in ranked
        )
        if ok:
            hits += 1
    return round(hits / len(test) * 100, 2) if len(test) else 0.0


def candidate_stats(preds: pd.DataFrame) -> dict:
    s = preds["n_candidates"]
    return {
        "mean_n_candidates": round(float(s.mean()), 2),
        "median_n_candidates": float(s.median()),
        "max_n_candidates": int(s.max()),
        "n_samples_with_candidates": int((s > 0).sum()),
        "n_samples_empty_pool": int((s == 0).sum()),
    }


def run_strategy(
    test: pd.DataFrame,
    long: pd.DataFrame,
    model_id: str,
    strategy: str,
    ts: str,
    *,
    recall_pathways: list[str] | None = None,
    recall_per_sample: bool = False,
) -> dict:
    preds = build_predictions(test, long, strategy)
    out_csv = OUT_DIR / f"predictions_{model_id}_{ts}.csv"
    preds.to_csv(out_csv, index=False)
    m = evaluate_preds(preds)
    pids = recall_pathways if recall_pathways is not None else pathway_ids_for_strategy(strategy)
    if strategy == "narrow":
        recall_per_sample = True
    r5 = recall_at_k(test, long, pids, k=5, per_sample_pathways=recall_per_sample)
    r5_ppm = recall_at_k_ppm(
        test, long, pids, k=5, per_sample_pathways=recall_per_sample
    )
    cov = candidate_stats(preds)
    pathways_label = preds["pathways_used"].iloc[0] if preds["pathways_used"].nunique() == 1 else "per-token (narrow)"
    row = {
        "model_or_tool": model_id,
        "baseline_family": _family(model_id),
        "setting": "ppm_rerank_top1_s2",
        "task": "A_targeted",
        "n_samples": len(test),
        "top_k": 1,
        "predictions_csv": str(out_csv.relative_to(REPO)),
        "pathways_used": pathways_label,
        **m,
        "recall_at_5_pct": r5,
        "recall_at_5_ppm_pct": r5_ppm,
        **cov,
    }
    print(
        f"{model_id}: validity={m['validity_pct']}% ppm={m['ppm_pass_pct']}% "
        f"exact={m['exact_match_pct']}% recall@5={r5}% mean_cand={cov['mean_n_candidates']}",
        flush=True,
    )
    return row


def _family(model_id: str) -> str:
    if model_id.startswith("v5"):
        return "sc_clm"
    if model_id.startswith("cts_"):
        return "cts"
    if "library" in model_id:
        return "library"
    return "biotransformer"


V5_PRED_FILES = {
    "v5b": OUT_DIR / "v5b_test_preds_cfg-on_tta-on_20260519.csv",
    "v5b1": OUT_DIR / "v5b1_test_preds_cfg-on_tta-on_20260518.csv",
    "v5a": OUT_DIR / "v5a_test_preds_cfg-on_tta-on_20260519.csv",
    "v5a1": OUT_DIR / "v5a1_test_preds_cfg-on_tta-on_20260518.csv",
}


def _latest_predictions_csv(model_id: str) -> Path | None:
    hits = sorted(OUT_DIR.glob(f"predictions_{model_id}_*.csv"))
    return hits[-1] if hits else None


def _pathway_to_model_id(pathway_id: str) -> str:
    if pathway_id == "library":
        return "patoon_library_s2"
    if pathway_id == "allHuman":
        return "bt_patoon_allHuman_s2"
    if pathway_id in CTS_PATHWAY_FILES:
        return f"cts_patoon_{pathway_id}_s2"
    return f"bt_patoon_{pathway_id}_s2"


def _metrics_row(
    token: str,
    n_samples: int,
    model_id: str,
    pathways_used: str,
    metrics: dict,
    *,
    recall_at_5: float | str = "",
    recall_at_5_ppm: float | str = "",
    cov: dict | None = None,
) -> dict:
    cov = cov or {}
    return {
        "token": token,
        "n_samples": n_samples,
        "model_or_tool": model_id,
        "baseline_family": _family(model_id),
        "pathways_used": pathways_used,
        "validity_pct": metrics["validity_pct"],
        "ppm_pass_pct": metrics["ppm_pass_pct"],
        "exact_match_pct": metrics["exact_match_pct"],
        "skeleton_pct": metrics["skeleton_pct"],
        "tanimoto_median": metrics["tanimoto_median"],
        "recall_at_5_pct": recall_at_5,
        "recall_at_5_ppm_pct": recall_at_5_ppm,
        "mean_n_candidates": cov.get("mean_n_candidates", ""),
        "median_n_candidates": cov.get("median_n_candidates", ""),
    }


def export_by_token_stratified(test: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    """按 test.token 分层：narrow + token 对应单途径 + V5A/B/A1/B1。"""
    rows: list[dict] = []
    narrow_path = _latest_predictions_csv("bt_patoon_narrow_s2")
    narrow_preds = pd.read_csv(narrow_path) if narrow_path else None

    v5_cache: dict[str, pd.DataFrame] = {}
    for branch, path in V5_PRED_FILES.items():
        if path.is_file():
            v5_cache[branch] = pd.read_csv(path)

    for token, grp in test.groupby("token", sort=True):
        n = len(grp)
        sids = set(grp["sample_id"])
        pathways = TOKEN_TO_BT_PATHWAYS.get(token, ["env"])
        pathways_label = ",".join(pathways)

        # narrow（复用全量预测或子集重算）
        if narrow_preds is not None:
            sub_pred = narrow_preds[narrow_preds["sample_id"].isin(sids)].copy()
        else:
            sub_pred = build_predictions(grp, long, "narrow")
        m = evaluate_preds(sub_pred)
        cov = candidate_stats(sub_pred)
        r5 = recall_at_k(grp, long, pathways, k=5, per_sample_pathways=True)
        r5p = recall_at_k_ppm(grp, long, pathways, k=5, per_sample_pathways=True)
        rows.append(
            _metrics_row(
                token,
                n,
                "bt_patoon_narrow_s2",
                pathways_label,
                m,
                recall_at_5=r5,
                recall_at_5_ppm=r5p,
                cov=cov,
            )
        )

        # token 映射到的各单途径
        for pid in pathways:
            mid = _pathway_to_model_id(pid)
            pred_path = _latest_predictions_csv(mid)
            if pred_path:
                sub_pred = pd.read_csv(pred_path)
                sub_pred = sub_pred[sub_pred["sample_id"].isin(sids)]
            else:
                sub_pred = build_predictions(grp, long, f"single:{pid}")
            m = evaluate_preds(sub_pred)
            cov = candidate_stats(sub_pred)
            r5 = recall_at_k(grp, long, [pid], k=5)
            r5p = recall_at_k_ppm(grp, long, [pid], k=5)
            rows.append(
                _metrics_row(
                    token,
                    n,
                    mid,
                    pid,
                    m,
                    recall_at_5=r5,
                    recall_at_5_ppm=r5p,
                    cov=cov,
                )
            )

        # SC-CLM 四分支（cfg-on + tta-on 预测）
        for branch in ("v5a", "v5b", "v5a1", "v5b1"):
            if branch not in v5_cache:
                continue
            vp = v5_cache[branch]
            merged = grp.merge(
                vp,
                on=["parent_smiles", "token", "delta_mz"],
                how="left",
                suffixes=("", "_v5"),
            )
            eval_df = merged.rename(columns={"target_smiles": "target_smiles"}).copy()
            if "prediction" not in eval_df.columns and "prediction_v5" in eval_df.columns:
                eval_df["prediction"] = eval_df["prediction_v5"]
            eval_df["prediction"] = eval_df["prediction"].fillna("")
            m = evaluate_preds(eval_df[["parent_smiles", "target_smiles", "prediction", "delta_mz"]])
            rows.append(
                _metrics_row(token, n, branch, "n/a", m),
            )

        narrow_ppm = next(
            r["ppm_pass_pct"]
            for r in rows
            if r["token"] == token and r["model_or_tool"] == "bt_patoon_narrow_s2"
        )
        v5b_ppm = next(
            (
                r["ppm_pass_pct"]
                for r in rows
                if r["token"] == token and r["model_or_tool"] == "v5b"
            ),
            "?",
        )
        v5a_ppm = next(
            (
                r["ppm_pass_pct"]
                for r in rows
                if r["token"] == token and r["model_or_tool"] == "v5a"
            ),
            "?",
        )
        print(
            f"  {token} n={n}: narrow={narrow_ppm}% v5b={v5b_ppm}% v5a={v5a_ppm}%",
            flush=True,
        )

    out = pd.DataFrame(rows)
    out_path = OUT_DIR / "summary" / "summary_external_patoon_by_token.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out)} rows)", flush=True)
    return out


def load_v5_reference() -> list[dict]:
    if not SUMMARY_MAIN.is_file():
        return []
    main = pd.read_csv(SUMMARY_MAIN)
    rows = []
    for branch in ("v5a", "v5b", "v5a1", "v5b1"):
        sub = main[(main["branch"] == branch) & (main["cfg"] == True) & (main["tta"] == True)]
        if sub.empty:
            continue
        r = sub.iloc[0]
        rows.append(
            {
                "model_or_tool": branch,
                "baseline_family": "sc_clm",
                "setting": "cfg-on_tta-on",
                "task": "A_targeted",
                "n_samples": int(r["n_samples"]),
                "top_k": "",
                "predictions_csv": f"(summary_main {branch})",
                "pathways_used": "n/a",
                "validity_pct": float(r["validity_pct"]),
                "ppm_pass_pct": float(r["ppm_pass_pct"]),
                "exact_match_pct": float(r["exact_match_pct"]),
                "skeleton_pct": float(r.get("skeleton_pct", float("nan"))),
                "tanimoto_median": float(r["tanimoto_median"]),
                "recall_at_5_pct": "",
                "recall_at_5_ppm_pct": "",
                "mean_n_candidates": "",
                "median_n_candidates": "",
                "max_n_candidates": "",
                "n_samples_with_candidates": "",
                "n_samples_empty_pool": "",
            }
        )
    return rows


def main() -> None:
    test = load_test()
    long = load_patoon_long()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    print(f"Loaded test n={len(test)}, patoon rows={len(long)}", flush=True)

    all_strategies: list[tuple[str, str]] = [
        ("bt_patoon_narrow_s2", "narrow"),
        ("bt_patoon_broad_s2", "broad"),
        ("cts_patoon_union_s2", "cts_union"),
    ]
    all_strategies.extend((mid, strat) for mid, strat, _ in COMPARISON_SINGLE_PATHWAYS)

    summary_rows: list[dict] = []

    for model_id, strategy in all_strategies:
        _, _, label = next(
            (t for t in COMPARISON_SINGLE_PATHWAYS if t[0] == model_id),
            (model_id, strategy, ""),
        )
        recall_p = [label] if label else pathway_ids_for_strategy(strategy)
        summary_rows.append(
            run_strategy(
                test,
                long,
                model_id,
                strategy,
                ts,
                recall_pathways=recall_p if recall_p and recall_p[0] else None,
                recall_per_sample=(strategy == "narrow"),
            )
        )

    summary_rows.extend(load_v5_reference())

    summary_dir = OUT_DIR / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    full_df = pd.DataFrame(summary_rows)
    out_summary = summary_dir / "summary_external_patoon.csv"
    full_df.to_csv(out_summary, index=False)

    # 逐项对照表：单途径 + library + CTS + V5 四分支
    comparison_ids = [mid for mid, _, _ in COMPARISON_SINGLE_PATHWAYS] + [
        "v5a",
        "v5b",
        "v5a1",
        "v5b1",
    ]
    comp_df = full_df[full_df["model_or_tool"].isin(comparison_ids)].copy()
    comp_df = comp_df.sort_values(
        by=["baseline_family", "ppm_pass_pct"],
        ascending=[True, False],
    )
    comp_path = summary_dir / "summary_external_patoon_comparison.csv"
    comp_df.to_csv(comp_path, index=False)

    cov_cols = [
        "model_or_tool",
        "recall_at_5_pct",
        "recall_at_5_ppm_pct",
        "mean_n_candidates",
        "median_n_candidates",
        "max_n_candidates",
        "n_samples_with_candidates",
        "n_samples_empty_pool",
    ]
    full_df[cov_cols].to_csv(
        summary_dir / "summary_external_patoon_coverage.csv", index=False
    )

    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "test_csv": str(TEST_CSV.relative_to(REPO)),
        "patoon_dir": str(PATOON_DIR.relative_to(REPO)),
        "n_test": len(test),
        "ppm_threshold": PPM_THRESHOLD,
        "comparison_csv": str(comp_path.relative_to(REPO)),
        "by_token_csv": "results/benchmark/v5/summary/summary_external_patoon_by_token.csv",
        "metrics": [
            "validity_pct",
            "ppm_pass_pct",
            "exact_match_pct",
            "skeleton_pct",
            "tanimoto_median",
            "recall_at_5_pct",
            "recall_at_5_ppm_pct",
            "mean_n_candidates",
        ],
        "note": "External baselines: Setting-2 PPM rerank top-1; recall@5 on PPM-ranked candidate pool.",
    }
    (summary_dir / "summary_external_patoon_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out_summary}", flush=True)
    print(f"Wrote {comp_path} ({len(comp_df)} rows)", flush=True)

    print("By-token stratified comparison...", flush=True)
    export_by_token_stratified(test, long)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--by-token-only",
        action="store_true",
        help="仅生成按 token 分层表（依赖已有 predictions_*.csv）",
    )
    args = parser.parse_args()
    if args.by_token_only:
        export_by_token_stratified(load_test(), load_patoon_long())
    else:
        main()
