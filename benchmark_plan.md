# SC-CLM V5 基准测试执行清单

本文档用于在训练结束后直接执行基准测试，覆盖 `CLAUDE.md` 中定义的 1-5 全量方案。

**执行进度快照（指标、产物路径、待办）**：[`results/benchmark/v5/BENCHMARK_PROGRESS.md`](results/benchmark/v5/BENCHMARK_PROGRESS.md) — 随跑次更新。

---

## 0. 统一原则（必须遵守）

- 固定评估脚本：`src/eval/metrics.py`
- 固定测试集：`data/processed/test.csv`（同一版本）
- 固定 PPM 口径：分母必须为 `target_mass`
- 固定后处理：SMILES 合法性检查、去重规则、候选截断规则一致
- 所有运行写入统一目录：`results/benchmark/v5/`
- 运行分工固定：本机执行 `V5A1/V5B1`（校准），服务器执行 `V5A/V5B`（正式）
- `V5A1/V5B1` 默认不进入正式 benchmark 主表，仅用于训练稳定性/吞吐校准对照
- 所有历史运行结果保留，禁止覆盖；新增运行必须使用新 `run_id`

---

## 1. 结果目录规范

建议目录结构：

```text
results/benchmark/v5/
  runs/
    <run_id>/
      predictions.csv
      metrics.json
      run_meta.json
  summary/
    summary_main.csv
    summary_seeds.csv
    summary_reliability.csv
    summary_efficiency.csv
    summary_external_patoon.csv              # 正式外部基线（全策略）
    summary_external_patoon_comparison.csv   # 逐项：BT 单途径 / CTS / library + V5
    summary_external_patoon_by_token.csv       # 按 token 分层 + V5A/B/A1/B1
    summary_external_patoon_coverage.csv     # recall@5、候选池统计
    summary_external.csv                     # 历史：Python JAR BT（探索性）
  patoon/
    patoon_bt_*_YYYYMMDD.csv
    patoon_cts_*_YYYYMMDD.csv
    patoon_library_default_YYYYMMDD.csv
  predictions_*_YYYYMMDD.csv
```

`run_id` 建议格式：

`<model>__cfg-<onoff>__tta-<onoff>__seed-<n>__k-<topk>__ts-<YYYYmmddHHMM>`

示例：

`v5a__cfg-on__tta-off__seed-42__k-5__ts-202605081630`

---

## 2. run_meta.json 字段模板（每次都写）

```json
{
  "run_id": "",
  "timestamp": "",
  "branch": "v5a|v5b|biotransformer|epa_cts",
  "checkpoint": "",
  "dataset": "data/processed/test.csv",
  "seed": 42,
  "top_k": 5,
  "cfg_enabled": true,
  "cfg_scale": null,
  "tta_enabled": false,
  "tta_n": 1,
  "decode_params": {
    "beam_size": null,
    "temperature": null,
    "top_p": null,
    "top_k_sampling": null
  },
  "postprocess": {
    "smiles_validity_check": true,
    "inchikey14_dedup": true
  },
  "code_version": {
    "git_commit": "",
    "metrics_script": "src/eval/metrics.py"
  },
  "hardware": {
    "gpu": "",
    "vram_gb": null
  }
}
```

---

## 3. 阶段A：主结论优先（1 + 2）

### 3.1 主评估集（1）

每次运行必须产出：

- `Validity`
- `Exact Match (InChIKey-14)`
- `PPM pass rate`
- `Tanimoto`

### 3.2 最小可比矩阵（2）

先跑 8 组（单 seed 初版）：

- 模型：`v5a`, `v5b`
- CFG：`off`, `on`
- TTA：`off`, `on`

共 `2 x 2 x 2 = 8` 组。

初版完成后，对 Top-2/Top-3 方案补 3 seeds。

---

## 4. 阶段B：稳健性与工程落地（3 + 4）

### 4.1 可靠性层（3）

对候选最优组新增三类子集评估：

- OOD scaffold 子集
- 长/复杂 SMILES 子集
- PPM 阈值边界子集

输出：

- 子集四指标（Validity/Exact/PPM/Tanimoto）
- 失败类型统计：`invalid`, `mass_mismatch`, `structure_mismatch`

### 4.2 工程效率层（4）

每个候选最优组记录：

- 吞吐：samples/s
- 平均延迟：ms/sample
- 显存峰值：GB
- 相对开销：相对 `cfg-off + tta-off` 的时间倍率

---

## 5. 外部基线层（5：BioTransformer + EPA CTS）

目标：回答“SC-CLM 是否优于外部工具”。

**当前状态（2026-05-24）**：阶段 E 已完成 **patRoon 正式基线**（Setting-2）；结论与路径见 [`results/benchmark/v5/BENCHMARK_PROGRESS.md`](results/benchmark/v5/BENCHMARK_PROGRESS.md) §0、§7。旧 Python `benchmark_bt.py` 结果保留于 `summary_external.csv`，**不作主结论**。

### 5.1 公平性约束（强制）

- 同一测试集：`data/processed/test.csv`
- 同一候选数口径：统一 top-k（建议 top-1 / top-5）
- 同一后处理与评估脚本：统一走 `src/eval/metrics.py`
- 同一 PPM 定义：`target_mass` 分母

### 5.1.1 任务定义与公平性协议（强制）

由于两类方法输入信息不同，必须拆成双任务评估，避免“单任务偏置结论”：

- **任务A：目标导向预测（Targeted Prediction）**
  - 输入：`parent SMILES + Δmz`
  - 输出：top-k 产物候选
  - 主要回答：给定观测质量信息时，谁能更准确命中目标产物

- **任务B：候选生成覆盖（Candidate Generation）**
  - 输入：`parent SMILES`
  - 输出：候选产物集合
  - 主要回答：不依赖质量条件时，谁的候选覆盖更完整

要求：任务A 与任务B都必须报告，不得只报其一。

### 5.1.2 三种信息设置（建议全报）

- **Setting-1（原生输入）**
  - SC-CLM：`SMILES + Δmz`
  - BioTransformer / CTS：`SMILES`
  - 用于评估真实应用场景下的端到端能力

- **Setting-2（统一质量重排，推荐主结论）**
  - BioTransformer / CTS 先生成候选，再按与观测 `Δmz` 的 ppm 偏差统一过滤/重排
  - SC-CLM 保持原生输入
  - 用于评估“在同等质量约束下的最终可用结果”

- **Setting-3（去质量信息能力下限）**
  - SC-CLM 不使用 `Δmz`（空值或默认占位）
  - BioTransformer / CTS 保持原生
  - 用于评估 SC-CLM 对质量条件信息的依赖程度

建议至少报告 Setting-1 + Setting-2；若篇幅允许补充 Setting-3。

### 5.2 输入输出映射约定

- BioTransformer / EPA CTS 原始输出先标准化为统一格式：
  - `sample_id`
  - `parent_smiles`
  - `pred_smiles`
  - `rank`
  - `score`（若有）
  - `source_tool`（`biotransformer` / `epa_cts`）
- 再由同一评估脚本计算四指标，避免“多口径”。

### 5.2.1 数据使用建议（按任务拆分）

- 任务A：使用配对真值集（`parent, product, Δmz`）
- 任务B：使用父体级集合（`parent`，并检查真值产物是否被候选覆盖）

其中任务B额外建议输出：

- `Hit@k / Recall@k`
- 候选数量（candidate set size）
- 有效率（valid candidate ratio）

### 5.3 结果汇总

**正式（patRoon）** — 由 `scratch/eval_patoon_external.py` 生成：

| 文件 | 内容 |
|------|------|
| `summary_external_patoon_comparison.csv` | allHuman / superbio / cyp450 / hgut / env / ecbased / library / CTS 分库 + V5A/B/A1/B1 |
| `summary_external_patoon_by_token.csv` | 每 `token`：narrow + 映射单途径 + V5 四分支 |
| `summary_external_patoon_coverage.csv` | `recall_at_5_pct`、`recall_at_5_ppm_pct`、`mean_n_candidates` 等 |

字段至少包含：`validity_pct`、`ppm_pass_pct`、`exact_match_pct`、`skeleton_pct`、`tanimoto_median`；外部行另含候选池指标。

**探索性（Python JAR）** — `summary_external.csv`：BT PPM ~31%，仅作历史对照。

**R 侧导出**：`scratch/patoon_tp_unify.R` → `results/benchmark/v5/patoon/patoon_*.csv`。

重跑命令：

```powershell
$env:PYTHONPATH = "<repo_root>"
.\venv\Scripts\python.exe scratch\eval_patoon_external.py
.\venv\Scripts\python.exe scratch\eval_patoon_external.py --by-token-only   # 仅分层表
```

---

## 6. 汇总表字段建议

`summary_main.csv`：

- run_id, branch, checkpoint, seed, cfg_enabled, tta_enabled, top_k
- validity, exact_match, ppm_pass_rate, tanimoto_mean, eval_runtime_sec

`summary_reliability.csv`：

- run_id, subset_name, validity, exact_match, ppm_pass_rate, tanimoto_mean
- invalid_count, mass_mismatch_count, structure_mismatch_count

`summary_efficiency.csv`：

- run_id, samples_per_sec, latency_ms_per_sample, peak_vram_gb, time_multiplier_vs_baseline

---

## 7. 执行顺序（冻结）

| 步骤 | 内容 | 状态 |
|------|------|------|
| 1 | 阶段 A：四分支 × CFG × TTA 矩阵 | ✅ |
| 2 | 多种子（V5B/V5B1） | ✅ |
| 3 | 阶段 B：可靠性 + 工程效率 | ✅ |
| 4 | 外部基线：patRoon BT/CTS/library（Setting-2） | ✅ |
| 4′ | 旧 Python BT | 探索性完成 |
| — | EPA CTS API | **不纳入**（CTS 经 patRoon BT 覆盖） |
| — | V5B 服务器训练 / 本机 V5A1·V5B1 LoRA | **训练已完成** |
| — | V5A2·V5B2 | **未执行**（由 A1/B1 覆盖，保留脚本入口） |
| 5 | 汇总 summary 并形成结论 | ✅（见 `BENCHMARK_PROGRESS.md`） |

---

## 8. 最终结论（2026-05-24 快照）

- **主结论（SC-CLM）**：**V5B / V5B1** 为可用主线（test PPM ~72%，Exact 最高 **30.28%** @ V5B1）；**V5A / V5A1** PPM 与 Exact 显著偏弱，不进入部署推荐。
- **稳健性**：OOD 接近全量；**长 SMILES** 为短板（PPM ~54–61%）；失败以 structure_mismatch 为主。
- **工程性**：cfg+tta best 相对 baseline **5–8×** 慢，显存仍低（~1.6 GB）。
- **外部对比（patRoon Setting-2）**：
  - **PPM**：superbio / narrow / allHuman 可 **高于 V5B**；hgut、ecbased、CTS 单库明显弱。
  - **Exact / Skeleton**：**V5B1 仍优于** 外部 BT top-1；library 在命中样本上 Exact 高但覆盖率低。
  - **分层**：氧化/脱甲基类 token 上 patRoon narrow 优势大；葡糖醛酸化、双氧化等 token 上 V5B 更优。
- **推荐配置**：推理 **V5B1 或 V5B**，`cfg-on` + `tta-on`，`beams=10`；外部对照表用 `summary_external_patoon_comparison.csv` + `by_token.csv`。

