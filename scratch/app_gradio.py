"""
SC-CLM Gradio 演示（MVP）

启动（仓库根目录）:
  pip install gradio
  set PYTHONPATH=<repo_root>
  python scratch/app_gradio.py
"""

from __future__ import annotations

import os
import sys
import io
from collections import Counter
from functools import lru_cache
from pathlib import Path

# 避免 Windows 代理 / Gradio 外连导致 launch 或 WebSocket 异常
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "0")
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
for _proxy in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_proxy, None)

import gradio as gr
import pandas as pd
from PIL import Image, ImageDraw
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdFMCS

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval.metrics import V4Metrics
from src.model.v5.config import V5Config
from src.model.v5.dataset import randomize_smiles
from src.model.v5.inference import V5Inference, get_mass

PPM_THRESHOLD = 10.0
TOKEN_TO_DELTA: dict[str, float] = dict(V5Config().token_to_delta)
DELTA_MATCH_TOL_DA = 0.05

CHECKPOINT_PRESETS: dict[str, str] = {
    "V5B1 LoRA（推荐）": "results/checkpoints/v5b1_lora_full_20260513_1029/best",
    "V5B 全量": "results/checkpoints/v5b/best",
    "V5A1 LoRA": "results/checkpoints/v5a1_lora_full_20260510_1127/best",
    "V5A 全量": "results/checkpoints/v5a/best",
}

TOKEN_CHOICES = list(TOKEN_TO_DELTA.keys())


def _resolve_delta(token: str, delta_mz: float | None) -> float:
    if delta_mz is None or delta_mz == "":
        return float(TOKEN_TO_DELTA[token])
    return float(delta_mz)


def _closest_token_for_delta(delta: float) -> tuple[str, float]:
    return min(TOKEN_TO_DELTA.items(), key=lambda kv: abs(kv[1] - delta))


def delta_hint(token: str, delta_mz: float | None) -> str:
    expected = TOKEN_TO_DELTA.get(token)
    if expected is None:
        return ""
    if delta_mz is None or delta_mz == "":
        return (
            f"ℹ️ `{token}` 默认 **Δm/z = {expected:.4f} Da**。"
            " 填写产物 Monoisotopic Mass 时将优先按质量推算 Δm/z。"
        )
    d = float(delta_mz)
    if abs(d - expected) <= DELTA_MATCH_TOL_DA:
        return f"✅ Δm/z 与 `{token}` 默认一致（**{expected:.4f} Da**）。"
    alt_t, alt_d = _closest_token_for_delta(d)
    return (
        f"⚠️ **Δm/z 与 token 不一致**：当前 **{d:.4f} Da**，"
        f"`{token}` 默认为 **{expected:.4f} Da**；"
        f"更接近 **`{alt_t}`**（{alt_d:.4f} Da）。"
    )


def _mass_errors(
    parent: str, pred: str, theoretical_delta: float
) -> tuple[float | None, float | None, float | None, float | None]:
    """返回 (abs_mass_error_Da, abs_mass_error_mDa, ppm, target_mass)。"""
    p_mass = get_mass(parent)
    pr_mass = get_mass(pred) if pred else None
    if not (p_mass and pr_mass):
        return None, None, None, None
    target_mass = p_mass + float(theoretical_delta)
    if target_mass <= 0:
        return None, None, None, None
    actual_delta = pr_mass - p_mass
    abs_da = abs(actual_delta - float(theoretical_delta))
    ppm = abs_da / target_mass * 1e6
    return abs_da, abs_da * 1000.0, ppm, target_mass


def _ppm_error(parent: str, pred: str, delta: float) -> float | None:
    _, _, ppm, _ = _mass_errors(parent, pred, delta)
    return ppm


def _rerank_ppm(parent: str, delta: float, candidates: list[str]) -> tuple[str, float | None]:
    metrics = V4Metrics()
    best_smi, best_ppm = "", None
    for c in candidates:
        c = (c or "").strip()
        if not c or not metrics.check_validity(c):
            continue
        pe = _ppm_error(parent, c, delta)
        if pe is None:
            continue
        if best_ppm is None or pe < best_ppm:
            best_ppm, best_smi = pe, c
    if best_smi:
        return best_smi, best_ppm
    for c in candidates:
        c = (c or "").strip()
        if c:
            return c, _ppm_error(parent, c, delta)
    return "", None


def _ppm_pass_column() -> str:
    return f"ppm_pass_{PPM_THRESHOLD:g}"


def _rank_candidates_for_display(
    parent: str,
    delta: float,
    candidates: list[str],
    target_smiles: str = "",
    limit: int | None = 10,
    only_ppm_pass: bool = False,
) -> pd.DataFrame:
    metrics = V4Metrics()
    target = (target_smiles or "").strip()
    ppm_col = _ppm_pass_column()
    seen: set[str] = set()
    rows = []

    for original_idx, candidate in enumerate(candidates):
        smi = (candidate or "").strip()
        if not smi:
            continue

        valid = metrics.check_validity(smi)
        key = metrics.canon(smi) if valid else f"invalid:{smi}"
        if key in seen:
            continue
        seen.add(key)

        abs_da, abs_mda, ppm, _ = _mass_errors(parent, smi, delta) if valid else (None, None, None, None)
        product_mass = get_mass(smi) if valid else None
        ppm_pass = bool(
            valid and metrics.check_ppm_fidelity(parent, smi, delta, threshold=PPM_THRESHOLD)
        )
        exact = metrics.canon(target) == metrics.canon(smi) if target and valid else None
        tanimoto = metrics.calc_tanimoto(target, smi) if target and valid else None
        rows.append({
            "_sort_ppm": ppm if ppm is not None else float("inf"),
            "_sort_idx": original_idx,
            "candidate_smiles": smi,
            "product_monoisotopic_mass_Da": product_mass,
            "abs_mass_error_Da": abs_da,
            "abs_mass_error_mDa": abs_mda,
            "ppm_error": ppm,
            ppm_col: ppm_pass,
            "validity": bool(valid),
            "exact_match": exact,
            "tanimoto": tanimoto,
        })

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows).sort_values(
        ["_sort_ppm", "_sort_idx"], ascending=[True, True]
    )
    if only_ppm_pass:
        table = table[table[ppm_col]]
    if limit is not None:
        table = table.head(limit)
    table = table.copy()
    table.insert(0, "rank", range(1, len(table) + 1))
    return table.drop(columns=["_sort_ppm", "_sort_idx"]).reset_index(drop=True)


def _collect_candidate_pool(
    engine: V5Inference,
    parent: str,
    model_token: str,
    use_cfg: bool,
    fast: bool,
    use_mass_mode: bool,
) -> tuple[list[str], float | None]:
    beam = 20 if use_mass_mode else 10
    if fast:
        return engine.predict_single(
            parent, model_token, num_beams=beam, num_return=beam, use_cfg=use_cfg
        ), None

    consensus, tta_conf, tta_candidates = _predict_tta_candidates(
        engine, parent, model_token, n_variants=10, use_cfg=use_cfg
    )
    direct = engine.predict_single(
        parent, model_token, num_beams=beam, num_return=beam, use_cfg=use_cfg
    )
    pool = ([consensus] if consensus else []) + list(tta_candidates) + list(direct)
    return pool, tta_conf


def _select_primary_prediction(
    ppm_pass_table: pd.DataFrame,
    top10: pd.DataFrame,
) -> str:
    if not ppm_pass_table.empty:
        return str(ppm_pass_table.iloc[0]["candidate_smiles"])
    if not top10.empty:
        return str(top10.iloc[0]["candidate_smiles"])
    return ""


def _predict_tta_candidates(
    engine: V5Inference,
    parent: str,
    token: str,
    n_variants: int,
    use_cfg: bool,
) -> tuple[str, float, list[str]]:
    all_candidates: list[str] = []
    for _ in range(n_variants):
        randomized = randomize_smiles(parent)
        all_candidates.extend(
            engine.predict_single(randomized, token, num_beams=5, num_return=5, use_cfg=use_cfg)
        )

    ik_counter: Counter = Counter()
    ik_to_smi: dict[str, str] = {}
    metrics = V4Metrics()
    for candidate in all_candidates:
        smi = (candidate or "").strip()
        if not smi:
            continue
        ik14 = metrics.get_inchikey_14(smi)
        if ik14:
            ik_counter[ik14] += 1
            ik_to_smi.setdefault(ik14, smi)

    if not ik_counter:
        return (all_candidates[0] if all_candidates else ""), 0.0, all_candidates

    top_ik, top_count = ik_counter.most_common(1)[0]
    return ik_to_smi[top_ik], top_count / len(all_candidates), all_candidates


def _mcs_bond_indices(mol: Chem.Mol, query: Chem.Mol, match: tuple[int, ...]) -> set[int]:
    bond_indices: set[int] = set()
    for bond in query.GetBonds():
        begin = match[bond.GetBeginAtomIdx()]
        end = match[bond.GetEndAtomIdx()]
        target_bond = mol.GetBondBetweenAtoms(begin, end)
        if target_bond is not None:
            bond_indices.add(target_bond.GetIdx())
    return bond_indices


def _draw_highlighted_mol(
    mol: Chem.Mol,
    legend: str,
    common_atoms: set[int],
    common_bonds: set[int],
    size: tuple[int, int] = (360, 340),
) -> Image.Image:
    changed_atoms = set(range(mol.GetNumAtoms())) - common_atoms
    changed_bonds = set(range(mol.GetNumBonds())) - common_bonds
    highlight_atoms = sorted(common_atoms | changed_atoms)
    highlight_bonds = sorted(common_bonds | changed_bonds)
    atom_colors = {
        **{idx: (0.35, 0.80, 0.35) for idx in common_atoms},
        **{idx: (1.00, 0.45, 0.35) for idx in changed_atoms},
    }
    bond_colors = {
        **{idx: (0.35, 0.80, 0.35) for idx in common_bonds},
        **{idx: (1.00, 0.45, 0.35) for idx in changed_bonds},
    }
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightBonds=highlight_bonds,
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    image = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")
    canvas = Image.new("RGB", (size[0], size[1] + 32), "white")
    canvas.paste(image, (0, 32))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), legend, fill="#333")
    return canvas


def _draw_mcs_change_highlight(parent_smi: str, product_smi: str) -> Image.Image | None:
    m_parent = Chem.MolFromSmiles(parent_smi)
    m_product = Chem.MolFromSmiles((product_smi or "").strip())
    if m_parent is None or m_product is None:
        return None

    mcs = rdFMCS.FindMCS(
        [m_parent, m_product],
        timeout=2,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
    )
    if mcs.numAtoms == 0 or not mcs.smartsString:
        return None

    query = Chem.MolFromSmarts(mcs.smartsString)
    if query is None:
        return None
    parent_match = m_parent.GetSubstructMatch(query)
    product_match = m_product.GetSubstructMatch(query)
    if not parent_match or not product_match:
        return None

    parent_common_atoms = set(parent_match)
    product_common_atoms = set(product_match)
    parent_common_bonds = _mcs_bond_indices(m_parent, query, parent_match)
    product_common_bonds = _mcs_bond_indices(m_product, query, product_match)

    size = (360, 340)
    parent_img = _draw_highlighted_mol(
        m_parent, "Parent: green=common, red=changed", parent_common_atoms, parent_common_bonds, size
    )
    product_img = _draw_highlighted_mol(
        m_product, "Prediction: green=common, red=changed", product_common_atoms, product_common_bonds, size
    )
    gap = 100
    legend_h = 42
    canvas = Image.new("RGB", (size[0] * 2 + gap, size[1] + 32 + legend_h), "white")
    canvas.paste(parent_img, (0, legend_h))
    canvas.paste(product_img, (size[0] + gap, legend_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "MCS common structure / changed region highlight", fill="#333")
    draw.text((size[0] + gap // 2 - 8, size[1] // 2 + legend_h + 20), "→", fill="black")
    return canvas


def _draw_transformation(parent_smi: str, product_smi: str) -> Image.Image | None:
    m_parent = Chem.MolFromSmiles(parent_smi)
    if m_parent is None:
        return None

    size = (340, 340)
    img_p = Draw.MolToImage(m_parent, size=size)
    p_mass = get_mass(parent_smi)

    product_smi = (product_smi or "").strip()
    m_prod = Chem.MolFromSmiles(product_smi) if product_smi else None
    if m_prod is None and product_smi:
        # 无效预测：只画母体 + 占位
        canvas_w = size[0] * 2 + 120
        canvas = Image.new("RGB", (canvas_w, size[1] + 48), "white")
        canvas.paste(img_p, (0, 40))
        draw = ImageDraw.Draw(canvas)
        draw.text((size[0] + 30, size[1] // 2 + 20), "→", fill="black")
        draw.text((10, 4), f"Parent  {p_mass:.4f} Da" if p_mass else "Parent", fill="#333")
        draw.text((size[0] + 120, 4), "Prediction invalid / empty", fill="#999")
        return canvas

    if m_prod is None:
        canvas = Image.new("RGB", (size[0], size[1] + 40), "white")
        canvas.paste(img_p, (0, 36))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 4), f"Parent  {p_mass:.4f} Da" if p_mass else "Parent", fill="#333")
        return canvas

    img_pred = Draw.MolToImage(m_prod, size=size)
    pr_mass = get_mass(product_smi)
    gap = 100
    canvas = Image.new("RGB", (size[0] * 2 + gap, size[1] + 48), "white")
    canvas.paste(img_p, (0, 40))
    canvas.paste(img_pred, (size[0] + gap, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text((size[0] + gap // 2 - 8, size[1] // 2 + 36), "→", fill="black")
    draw.text((8, 4), f"Parent  {p_mass:.4f} Da" if p_mass else "Parent", fill="#1a5276")
    draw.text(
        (size[0] + gap + 8, 4),
        f"Product  {pr_mass:.4f} Da" if pr_mass else "Product",
        fill="#117a65",
    )
    return canvas


@lru_cache(maxsize=2)
def _load_engine(checkpoint_rel: str, guidance: float) -> V5Inference:
    ckpt = REPO / checkpoint_rel
    if not (ckpt / "model.safetensors").is_file() and not (ckpt / "pytorch_model.bin").is_file():
        raise FileNotFoundError(f"未找到权重: {ckpt}")
    return V5Inference(str(ckpt), guidance_scale=guidance)


def _validate_parent(smiles: str) -> str | None:
    s = (smiles or "").strip()
    if not s:
        return "请输入 parent SMILES"
    if Chem.MolFromSmiles(s) is None:
        return "parent SMILES 无法被 RDKit 解析"
    return None


def sync_from_mass(parent_smiles: str, product_mass: float | None) -> tuple[str, float | None, str, str]:
    """由母体 SMILES + 产物 Monoisotopic Mass 推算 Δm/z 与 token。"""
    err = _validate_parent(parent_smiles)
    if err:
        return err, None, TOKEN_CHOICES[0], delta_hint(TOKEN_CHOICES[0], None)

    parent = parent_smiles.strip()
    p_mass = get_mass(parent)
    if p_mass is None:
        return "无法计算母体 Monoisotopic Mass", None, TOKEN_CHOICES[0], ""

    if product_mass is None or product_mass == "":
        return (
            f"母体 Monoisotopic Mass：**{p_mass:.4f} Da** — 请填写 **产物 Monoisotopic Mass** 后预测。",
            None,
            TOKEN_CHOICES[0],
            "",
        )

    t_mass = float(product_mass)
    delta = t_mass - p_mass
    token, token_delta = _closest_token_for_delta(delta)
    info = (
        f"母体 Monoisotopic Mass：**{p_mass:.4f} Da**  \n"
        f"目标产物 Monoisotopic Mass：**{t_mass:.4f} Da**  \n"
        f"推算 **Δm/z = {delta:.4f} Da** → 模型 token **`{token}`**"
        f"（库内最近 {token_delta:.4f} Da）"
    )
    return info, delta, token, delta_hint(token, delta)


def on_token_change(token: str) -> tuple[float, str]:
    d = TOKEN_TO_DELTA[token]
    return d, delta_hint(token, d)


def on_delta_change(token: str, delta_mz: float | None) -> str:
    return delta_hint(token, delta_mz)


def reset_delta_to_token_default(token: str) -> tuple[float, str]:
    d = TOKEN_TO_DELTA[token]
    return d, delta_hint(token, d)


def _resolve_inputs(
    parent_smiles: str,
    product_mass: float | None,
    token: str,
    delta_mz: float | None,
    use_mass_mode: bool,
) -> tuple[str | None, str, float, float | None, str]:
    """返回 (error, parent, delta, target_mass, token_for_model)。"""
    err = _validate_parent(parent_smiles)
    if err:
        return err, "", 0.0, None, token

    parent = parent_smiles.strip()
    p_mass = get_mass(parent)
    if p_mass is None:
        return "无法计算母体质量", parent, 0.0, None, token

    if use_mass_mode:
        if product_mass is None or product_mass == "":
            return "请填写产物 Monoisotopic Mass（Da）", parent, 0.0, None, token
        t_mass = float(product_mass)
        delta = t_mass - p_mass
        model_token, _ = _closest_token_for_delta(delta)
        return None, parent, delta, t_mass, model_token

    delta = _resolve_delta(token, delta_mz)
    t_mass = p_mass + delta
    return None, parent, delta, t_mass, token


def predict_single(
    parent_smiles: str,
    product_mass: float | None,
    use_mass_mode: bool,
    token: str,
    delta_mz: float | None,
    mode: str,
    checkpoint_preset: str,
    use_cfg: bool,
    guidance: float,
    target_smiles: str,
) -> tuple[str, str, pd.DataFrame, pd.DataFrame, pd.DataFrame, Image.Image | None, Image.Image | None]:
    err, parent, delta, target_mass, model_token = _resolve_inputs(
        parent_smiles, product_mass, token, delta_mz, use_mass_mode
    )
    if err:
        return err, "", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, None

    hint = delta_hint(model_token if not use_mass_mode else token, delta)
    mismatch = (not use_mass_mode) and hint.startswith("⚠️")
    ckpt_rel = CHECKPOINT_PRESETS[checkpoint_preset]

    try:
        engine = _load_engine(ckpt_rel, float(guidance))
    except Exception as e:
        return f"模型加载失败: {e}", "", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, None

    fast = mode.startswith("快速")
    tta_conf = None
    ppm_pass_table = pd.DataFrame()

    try:
        pool, tta_conf = _collect_candidate_pool(
            engine, parent, model_token, use_cfg, fast, use_mass_mode
        )
        top10 = _rank_candidates_for_display(parent, delta, pool, target_smiles, limit=10)
        if use_mass_mode:
            ppm_pass_table = _rank_candidates_for_display(
                parent, delta, pool, target_smiles, limit=None, only_ppm_pass=True
            )
            pred = _select_primary_prediction(ppm_pass_table, top10)
        else:
            pred = _select_primary_prediction(pd.DataFrame(), top10)
    except Exception as e:
        return f"推理失败: {e}", "", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, None

    metrics = V4Metrics()
    valid = metrics.check_validity(pred) if pred else False
    abs_da, abs_mda, ppm, _ = _mass_errors(parent, pred, delta)
    ppm_pass = (
        metrics.check_ppm_fidelity(parent, pred, delta, threshold=PPM_THRESHOLD)
        if pred and valid
        else False
    )

    struct_img = _draw_transformation(parent, pred)
    change_img = _draw_mcs_change_highlight(parent, pred)

    lines = []
    if mismatch:
        lines.append(hint)
    if use_mass_mode:
        lines.append(
            f"**输入模式**：母体 SMILES + 产物 Monoisotopic Mass（目标 **{target_mass:.4f} Da**）"
        )
        lines.append(
            f"**PPM≤{PPM_THRESHOLD:g} 合格候选数**: {len(ppm_pass_table)}"
            + ("（主预测取自合格候选中 PPM 最小者）" if not ppm_pass_table.empty else "（无合格候选，主预测回退为 PPM 最优）")
        )
    lines.extend([
        f"**预测产物**: `{pred or '(空)'}`",
        f"**模式**: {mode} | **checkpoint**: `{ckpt_rel}`",
        f"**模型 token**: `{model_token}` | **Δm/z（质量约束）**: {delta:.4f} Da",
    ])
    if abs_da is not None:
        lines.append(
            f"**绝对质量误差**: **{abs_da:.6f} Da**（**{abs_mda:.3f} mDa**）"
            + (f" | **PPM 误差**: {ppm:.2f}" if ppm is not None else "")
        )
    lines.append(
        f"**合法性**: {'通过' if valid else '未通过'} | **PPM≤{PPM_THRESHOLD:g}**: {'通过' if ppm_pass else '未通过'}"
    )
    if tta_conf is not None:
        lines.append(f"**TTA 共识置信度**: {tta_conf:.2%}")
    if (target_smiles or "").strip():
        t = target_smiles.strip()
        lines.append(
            f"**相对真值 Exact**: {'是' if metrics.canon(t) == metrics.canon(pred) else '否'}"
        )
        lines.append(f"**Tanimoto**: {metrics.calc_tanimoto(t, pred):.4f}")

    table = pd.DataFrame(
        [{"字段": k, "值": v} for k, v in [
            ("parent_smiles", parent),
            ("parent_monoisotopic_mass_Da", get_mass(parent)),
            ("target_product_monoisotopic_mass_Da", target_mass),
            ("model_token", model_token),
            ("delta_mz", delta),
            ("prediction", pred),
            ("abs_mass_error_Da", abs_da),
            ("abs_mass_error_mDa", abs_mda),
            ("ppm_error", ppm),
            ("validity", valid),
            ("ppm_pass", ppm_pass),
        ]]
    )
    return "\n\n".join(lines), pred or "", table, top10, ppm_pass_table, struct_img, change_img


def predict_batch(
    file_obj,
    mode: str,
    checkpoint_preset: str,
    use_cfg: bool,
    guidance: float,
    progress=gr.Progress(),
):
    if file_obj is None:
        return None, "请上传 CSV 文件"

    path = Path(file_obj.name if hasattr(file_obj, "name") else file_obj)
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return None, f"读取 CSV 失败: {e}"

    for col in ("parent_smiles", "token", "delta_mz"):
        if col not in df.columns:
            return None, f"CSV 缺少列: {col}（需要 parent_smiles, token, delta_mz）"

    ckpt_rel = CHECKPOINT_PRESETS[checkpoint_preset]
    try:
        engine = _load_engine(ckpt_rel, float(guidance))
    except Exception as e:
        return None, f"模型加载失败: {e}"

    fast = mode.startswith("快速")
    out_rows = []
    n = len(df)
    for i, row in df.iterrows():
        progress((i + 1) / n, desc=f"{i + 1}/{n}")
        try:
            if fast:
                parent = str(row["parent_smiles"])
                token = str(row["token"])
                delta = float(row["delta_mz"])
                cands = engine.predict_single(
                    parent, token, num_beams=3, num_return=3, use_cfg=use_cfg
                )
                pred, ppm = _rerank_ppm(parent, delta, cands)
                abs_da, abs_mda, _, _ = _mass_errors(parent, pred, delta)
                out_rows.append({
                    **row.to_dict(),
                    "prediction": pred,
                    "abs_mass_error_Da": abs_da,
                    "abs_mass_error_mDa": abs_mda,
                    "ppm_error": ppm,
                    "tta_confidence": None,
                })
            else:
                part = engine.predict_batch_with_rerank(
                    pd.DataFrame([row]), use_cfg=use_cfg, tta=10, num_beams=10
                )
                out_rows.append(part.iloc[0].to_dict())
        except Exception as e:
            out_rows.append({**row.to_dict(), "prediction": "", "error": str(e)})

    out = pd.DataFrame(out_rows)
    out_path = REPO / "results" / "benchmark" / "v5" / "gradio_batch_latest.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return str(out_path), f"完成 {n} 条，已保存至 `{out_path.relative_to(REPO)}`"


def preload_default_model() -> str:
    try:
        ckpt = CHECKPOINT_PRESETS["V5B1 LoRA（推荐）"]
        _load_engine(ckpt, 1.5)
        return f"默认模型已加载：`{ckpt}`"
    except Exception as e:
        return f"预加载失败（首次预测时会再试）：{e}"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="SC-CLM V5 预测") as demo:
        gr.Markdown(
            "# SC-CLM 环境转化产物预测\n"
            "**推荐输入**：母体 SMILES + **产物 Monoisotopic Mass（Da）** → 自动推算 Δm/z 并预测结构。"
            " 推荐模型：**V5B1 LoRA**。"
        )
        load_status = gr.Markdown("正在预加载模型…")

        with gr.Row():
            checkpoint = gr.Dropdown(
                list(CHECKPOINT_PRESETS.keys()),
                value="V5B1 LoRA（推荐）",
                label="模型",
            )
            mode = gr.Radio(
                ["快速（少 beam，无 TTA）", "高精度（TTA+CFG，慢）"],
                value="快速（少 beam，无 TTA）",
                label="速度档位",
            )

        with gr.Tab("单条预测"):
            use_mass_mode = gr.Checkbox(
                value=True,
                label="按 Monoisotopic Mass 预测（推荐）",
                info="勾选：只需母体 SMILES + 产物质量；取消：手动选 token / Δm/z",
            )
            with gr.Row():
                parent = gr.Textbox(
                    label="母体 Parent SMILES",
                    lines=3,
                    placeholder="例如 CCNC1=NC(NC(C)C)=NC(Cl)=N1",
                )
                with gr.Column():
                    product_mass = gr.Number(
                        label="产物 Monoisotopic Mass (Da)",
                        precision=6,
                        info="目标产物的精确单同位素质量；与母体质量之差即 Δm/z",
                    )
                    mass_info_md = gr.Markdown("填写母体 SMILES 与产物 Monoisotopic Mass。")

            with gr.Accordion("Token / Δm/z 高级（可选）", open=False):
                token = gr.Dropdown(
                    TOKEN_CHOICES,
                    value="[TRANS_OXIDATION]",
                    label="转化类型 (token)",
                )
                delta = gr.Number(
                    label="Δm/z (Da)",
                    value=TOKEN_TO_DELTA["[TRANS_OXIDATION]"],
                    precision=4,
                )
                delta_hint_md = gr.Markdown(
                    delta_hint("[TRANS_OXIDATION]", TOKEN_TO_DELTA["[TRANS_OXIDATION]"])
                )
                reset_delta_btn = gr.Button("恢复 token 默认 Δm/z", size="sm")

            target = gr.Textbox(
                label="真值产物 SMILES（可选，仅用于结构/指标对比）",
                lines=2,
            )

            parent.change(sync_from_mass, [parent, product_mass], [mass_info_md, delta, token, delta_hint_md])
            product_mass.change(
                sync_from_mass, [parent, product_mass], [mass_info_md, delta, token, delta_hint_md]
            )
            token.change(on_token_change, [token], [delta, delta_hint_md])
            delta.change(on_delta_change, [token, delta], [delta_hint_md])
            reset_delta_btn.click(reset_delta_to_token_default, [token], [delta, delta_hint_md])

            with gr.Accordion("高级选项", open=False):
                use_cfg = gr.Checkbox(value=True, label="CFG")
                guidance = gr.Slider(1.0, 2.5, value=1.5, step=0.1, label="CFG guidance")

            btn = gr.Button("预测", variant="primary")
            summary = gr.Markdown()
            with gr.Row():
                pred_out = gr.Textbox(label="预测 SMILES", lines=2, scale=1)
                struct_img = gr.Image(label="母体 → 预测产物 结构", type="pil", scale=1)
            change_img = gr.Image(label="公共结构 / 变化区域（MCS 高亮）", type="pil")
            detail = gr.Dataframe(label="明细")
            ppm_pass_df = gr.Dataframe(
                label=f"PPM≤{PPM_THRESHOLD:g} 合格候选（质量模式，全部列出）"
            )
            top10 = gr.Dataframe(label="Top10 候选（按 PPM / 质量误差排序，含未通过）")

            btn.click(
                predict_single,
                [
                    parent,
                    product_mass,
                    use_mass_mode,
                    token,
                    delta,
                    mode,
                    checkpoint,
                    use_cfg,
                    guidance,
                    target,
                ],
                [summary, pred_out, detail, top10, ppm_pass_df, struct_img, change_img],
                queue=True,
            )

        with gr.Tab("批量 CSV"):
            gr.Markdown(
                "上传 CSV，列名需含 **`parent_smiles`**, **`token`**, **`delta_mz`**。"
            )
            batch_file = gr.File(label="CSV", file_types=[".csv"])
            batch_btn = gr.Button("批量预测", variant="primary")
            batch_status = gr.Markdown()
            batch_dl = gr.File(label="下载结果")

            batch_btn.click(
                predict_batch,
                [batch_file, mode, checkpoint, use_cfg, guidance],
                [batch_dl, batch_status],
                queue=True,
            )

        gr.Markdown(
            "---\n"
            f"PPM 阈值 **{PPM_THRESHOLD:g} ppm**（分母 target_mass = 母体质量 + Δm/z）。"
            " 绝对质量误差 = |预测产物质量 − 目标产物 Monoisotopic Mass|。"
        )
        demo.load(preload_default_model, outputs=load_status)
    return demo


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--no-preload", action="store_true")
    args = parser.parse_args()

    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        show_error=True,
        inbrowser=False,
    )


if __name__ == "__main__":
    main()
