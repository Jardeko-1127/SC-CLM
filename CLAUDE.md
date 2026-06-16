# CLAUDE.md

本文件是我与 Claude Code（CC）协作的主窗口。  
**从现在开始，所有交流与输出均使用中文。**

---

## 项目概览

SC-CLM 用于环境转化产物预测，核心任务是：

`parent SMILES -> product SMILES`

当前 V5 采用双分支：

- **V5A（主线）**：MolT5-Small + embedding-conditioned encoder-decoder + CFG
- **V5B（实验线）**：ReactionT5v2-forward + embedding-conditioned encoder-decoder + CFG
- **V5A1（低显存 LoRA 线）**：MolT5-Small + LoRA（attention-only）+ 低显存训练配置；**默认按 epoch 完整 LoRA**（可用环境变量改短跑校准）
- **V5B1（低显存 LoRA 线）**：ReactionT5v2-forward + LoRA（attention-only）+ 低显存训练配置；**默认按 epoch 完整 LoRA**（可用环境变量改短跑校准）
- **V5A2 / V5B2（保留入口，未执行）**：原拟服务器 LoRA 独立目录；后确认本机 **V5A1/V5B1** 即可跑通完整 LoRA，**不再另训 A2/B2**。脚本 `train_v5a2.py` / `train_v5b2.py` 仍可用于将来需隔离目录时手动启动。

V4/V3 仅作历史参考（见 `ref/`）。

### V6 新一代架构（2026-06-15 实现）

V6 将 V5 的离散反应 token 替换为 **连续质量差 (delta_mz) 量化嵌入**，实现 `parent SMILES + delta_mz → product SMILES`。

- **V6a（主线）**：ReactionT5v2-forward + 532-bin Mass Embedding + CFG + LoRA（V5B1 风格）
- **V6b（对照线）**：OpenNMT Transformer 6L + Mass Token Prefix + Beam Search

| 维度 | V6a | V6b |
|---|---|---|
| Backbone | ReactionT5v2 (~80M) | Transformer 6L (~45M) |
| 条件注入 | Embedding prepend (768-dim) | Input token prefix (MASS_X) |
| 分词 | BPE subword | Character-level (Kekulé) |
| 推理 | CFG (γ=1.5) | Beam search (b=20) |
| 框架 | transformers + peft | opennmt-py + ctranslate2 |
| 数据 | raw+CTS 合并 ~12K pairs | 同 V6a，转 OpenNMT 格式 |
| 测试 | `tests/test_v6a.py` (26) | `tests/test_v6b.py` (15) |

**质量量化方案**（V6a/V6b 共享 532 bins）：
- 核心区 [-100, 200] @1 Da → 300 bins
- 外围 [-500, -100) ∪ [200, 500] @5 Da → 140 bins
- 极值 (< -500 或 ≥ 500) @10 Da → 92 bins

**已知限制**：opennmt-py 要求 torch<2.3，与 transformers 要求的 torch≥2.4 冲突。V6b 训练需独立 conda 环境。

---

## 环境与执行

```bash
# Windows PowerShell
.\venv\Scripts\python.exe <script>

# Windows Git Bash
source venv/Scripts/activate && python <script>
```

Python 3.12.10，依赖在 `venv/` 管理。

### 运行分工（当前固定）

- **本机（当前电脑）**：代码/测试/数据维护；快速验证可 `$env:V6_NUM_EPOCHS='2'; $env:V6_MICRO_BATCH='1'` 跑冒烟
- **服务器**：执行并保留 **`V5A / V5B`** + **`V6a`**；V6a 通过 `SC_CLM_SERVER=1` 启用服务端配置
- **V5A2 / V5B2**：**不跑**（原计划服务器 LoRA，后由本机 A1/B1 覆盖需求）
- **V6b**：待独立环境就绪后部署（opennmt-py torch<2.3 冲突）
- **保存策略**：所有日志与 checkpoint **全部保留**，禁止覆盖旧结果；新一轮运行请使用独立目录或时间戳后缀

---

## 常用命令

```bash
# 1) 数据预处理
python src/data/preprocess.py

# 2) 训练
python src/model/v5/train_v5a.py   # 主线A
python src/model/v5/train_v5b.py   # 主线B
python src/model/v5/train_v5a1.py  # 低显存 LoRA（MolT5）；默认完整 LoRA，见 V5B1_MAX_STEPS / V5B1_NUM_EPOCHS
python src/model/v5/train_v5b1.py  # 低显存 LoRA（ReactionT5）；同上
# train_v5a2.py / train_v5b2.py  # 可选：独立 v5a2/v5b2 目录；当前项目未执行

# 3) 推理
python src/model/v5/inference.py

# 4) 评估
python src/eval/metrics.py
python src/eval/error_analysis.py
python src/eval/benchmark_bt.py
python src/eval/benchmark_oracle.py

# 5) V6 数据准备
python src/model/v6/prepare_data.py        # V6a CSV 数据（raw+CTS 合并）
python src/model/v6b/prepare_data.py        # V6b OpenNMT 格式数据
python src/model/v6b/build_vocab.py         # V6b 词汇表构建

# 6) V6 训练
python src/model/v6/train.py                # V6a LoRA 训练
python src/model/v6b/train.py               # V6b OpenNMT 训练（需独立 torch<2.3 环境）

# 7) V6 推理
python src/model/v6/inference.py
python src/model/v6b/inference.py

# 8) V6 评估
python src/model/v6b/evaluate.py            # V6a vs V6b 同口径对比

# 9) V6 测试
python tests/test_v6a.py                    # 26 项
python tests/test_v6b.py                    # 15 项
```

---

## 当前训练状态（以最新日志为准）


| 分支   | 基座                            | 状态                            | 主日志                   | checkpoint 目录                     | 最佳指标                 |
| ---- | ----------------------------- | ----------------------------- | --------------------- | --------------------------------- | -------------------- |
| V5A  | `laituan245/molt5-small`      | **已完成**（60/60 epoch，步数 51900） | `logs/v5a/train.log`  | `results/checkpoints/v5a/`        | `eval_ppm_pass_rate` |
| V5B  | `sagawa/ReactionT5v2-forward` | **已完成**（服务器，60 epoch）；本机保留 `v5b/best` 推理包 | 服务器 `logs/v5b/`；benchmark 用本机 `v5b/best` | `results/checkpoints/v5b/best/`（tar 同步） | `eval_ppm_pass_rate` |
| V5A1 | `laituan245/molt5-small`      | **默认完整 LoRA**（`V5B1_MAX_STEPS=-1`，60 epoch）；历史：300-step 校准已跑通 | `logs/v5a1/*.log`（建议时间戳命名） | `results/checkpoints/v5a1_*`（独立目录） | 完整 LoRA 以 `ppm_pass_rate` 选优；短跑无 epoch eval |
| V5B1 | `sagawa/ReactionT5v2-forward` | **默认完整 LoRA**（同上）；历史：300-step 校准已跑通                         | `logs/v5b1/*.log`（建议时间戳命名） | `results/checkpoints/v5b1_*`（独立目录） | 同上 |
| V5A2 | `laituan245/molt5-small`      | **未执行**（由 V5A1 覆盖） | — | — | 保留 `train_v5a2.py` 入口 |
| V5B2 | `sagawa/ReactionT5v2-forward` | **未执行**（由 V5B1 覆盖） | — | — | 保留 `train_v5b2.py` 入口 |
| V6a  | `sagawa/ReactionT5v2-forward` | **服务器训练中**（`SC_CLM_SERVER=1`） | 服务器 `logs/v6a/` | `results/checkpoints/v6a/` | 待服务器回报 |
| V6b  | Transformer 6L (~45M)        | **未训练**（需独立 torch<2.3 环境） | — | `results/checkpoints/v6b/` | 待环境就绪 |


V5A 口径：

- 启动时间：2026-05-07 19:22（最终一次成功启动）
- 结束时间：2026-05-08 17:26，总耗时 ~22h
- `augment_n = 5`，bf16，effective batch=32，CFG dropout=0.15
- 训练集 5530 base × 5 aug = 27650 effective；验证集 399
- **最佳 checkpoint：`checkpoint-49305`（epoch 57）**，`best_metric=19.3`
- 最佳指标：`**eval_ppm_pass_rate = 19.3%`**，`eval_loss = 0.54893`，`eval_exact_match = 5.01%`，`eval_validity = 70.18%`
- 末点指标（checkpoint-51900，epoch 60）：`eval_ppm_pass_rate = 17.04%`，`eval_loss = 0.55150`（已过拟合）
- 每个 `checkpoint-*` 下含 `trainer_state.json`
- 训练最佳点以 `trainer_state.json` 的 `best_metric` / `best_model_checkpoint` 为准

V5B 训练超参口径（以 `src/model/v5/train_v5b.py` + `src/model/v5/config.py` 为准）：

- Backbone：`sagawa/ReactionT5v2-forward`（建议用本地 `models/ReactionT5v2-forward`）
- `reaction_embed_dim = 768`（必须与 backbone `d_model=768` 一致）
- `augment_n = 5`
- 有效 batch：`per_device_train_batch_size=4`，`gradient_accumulation_steps=8` → **effective batch=32**
- 精度：优先 bf16（支持则开），否则 fp16（脚本自动选择）
- CFG（训练）：`cfg_dropout_prob = 0.15`
- 学习率：`learning_rate = 2e-4`
- warmup：`warmup_ratio = 0.1`
- 训练轮数：`num_epochs = 60`
- 序列长度：`max_input_len = 512`，`max_output_len = 512`
- 评估/保存：`eval_strategy="epoch"`，`save_strategy="epoch"`，`save_total_limit=3`
- 最佳模型选择：`load_best_model_at_end=True`，`metric_for_best_model="ppm_pass_rate"`，`greater_is_better=True`
- 早停：`patience = 15`（EarlyStoppingCallback）
- 生成评估：`predict_with_generate=True`，`generation_num_beams=1`，`generation_max_length=max_output_len`
- DataLoader：`dataloader_num_workers=0`
- 日志与产物：`logs/v5b/train.log`，`results/checkpoints/v5b/`

V5A1/V5B1/V5A2/V5B2（LoRA 低显存线）统一口径（以 `src/model/v5/train_v5b1.py` 为内核；`train_v5a1.py` / `train_v5a2.py` / `train_v5b2.py` 为入口）：

- LoRA：`r=16`，`alpha=32`，`target_modules=[q,k,v,o]`（仅 attention 层）
- 低显存配置：`micro_batch=1`，`gradient_accumulation=4~8`（当前默认 8）
- 序列长度：MolT5=`128/128`，ReactionT5=`256/128`
- Gradient checkpointing：开启（在 `base_model` 手动启用）
- 默认训练精度：
  - V5B1（ReactionT5）：`fp16=True`（`V5B1_FORCE_FP16`）
  - V5A1（MolT5）：默认 `fp16=False`（用于规避 `loss=0 / grad_norm=nan / lr=0` 数值异常）
- **训练长度与环境变量（`train_v5b1.py` 内核）**
  - **`V5B1_MAX_STEPS`**：默认 **`-1`** → **完整 LoRA**：`Trainer` 使用 `max_steps=-1`，按 **`V5Config.num_epochs`（默认 60）** 跑 epoch。
  - 设为正整数（例如 **`300`**）→ **短跑校准**：固定步数结束；关闭按 epoch 的 eval/checkpoint（仅冒烟、吞吐或快速诊断）。
  - **`V5B1_NUM_EPOCHS`**：若设置（非空），覆盖 `num_epochs`（例如本机先试 `10`）。
- **完整 LoRA 时的训练行为**：每 epoch **eval + save**，`predict_with_generate=True`，`load_best_model_at_end=True`，`metric_for_best_model="ppm_pass_rate"`，`greater_is_better=True`，早停 **`patience=15`**（与 `V5Config` 一致），`save_total_limit=3`；训练结束后额外落盘 **`{output_dir}/best/`**。
- 训练脚本：
  - 内核：`src/model/v5/train_v5b1.py`
  - MolT5 入口：`src/model/v5/train_v5a1.py`（本机常用）、`train_v5a2.py`（服务器：独立 `v5a2` 目录与 `_status`）
  - ReactionT5 入口：`train_v5b1.py`、`train_v5b2.py`（服务器：独立 `v5b2`）
  - 可选覆盖：`V5_LORA_BRANCH_DISPLAY`（日志/状态里显示的分支名；V5A2/V5B2 入口已自动设置）

V6a 训练口径（以 `src/model/v6/train.py` 为准）：

- Backbone：`sagawa/ReactionT5v2-forward`（优先本地 `models/ReactionT5v2-forward`）
- Mass bins：**532**（核心 1Da / 外围 5Da / 极值 10Da）
- `mass_embed_dim = 768`（必须匹配 backbone `d_model=768`）
- LoRA：`r=16`，`alpha=32`，`dropout=0.1`，`target_modules=[q,k,v,o]`（仅 attention）
- CFG（训练）：`cfg_dropout_prob = 0.15`；推理：`cfg_guidance_scale = 1.5`
- **本机模式**（默认）：`micro_batch=1`，`grad_accum=8` → effective batch=8；`gradient_checkpointing=True`
- **服务器模式**（`SC_CLM_SERVER=1`）：可配 `V6_MICRO_BATCH`(4)、`V6_GRAD_ACCUM`(4)、`V6_NUM_EPOCHS`(20)、`V6_PATIENCE`(5)、`V6_DATALOADER_WORKERS`(4)；关 checkpointing
- 学习率：`2e-4`，warmup：0.1，epoch：30（本机）/ 20（服务器），patience：10（本机）/ 5（服务器）
- 序列长度：`max_input_len=256`，`max_output_len=128`
- 增强：`augment_n=5`（SMILES randomization）
- 最佳模型选择：`load_best_model_at_end=True`，`metric_for_best_model="ppm_pass_rate"`
- 日志与产物：`logs/v6a/`，`results/checkpoints/v6a/`

V6a Checkpoint 落盘格式（与 V5 对齐）：
- `pytorch_model.bin`：完整 `V6SCLM.state_dict()`
- `v6_state.pt`：`mass_embed`、`cfg_*`、`num_mass_bins`、`mass_embed_dim`、`mass_bins`
- tokenizer 文件 + `training_args.bin`
- 训练结束后 `merge_and_unload()` LoRA → `{output_dir}/best/`

V6b 训练口径（以 `src/model/v6b/train.py` + `src/model/v6b/config.py` 为准）：

- 框架：OpenNMT-py（`onmt_train` CLI），需独立 conda 环境（torch<2.3）
- Backbone：Transformer 6L/8H/512d/2048ff，~45M 参数
- 分词：字符级（Kekulé SMILES，空格分隔），质量 token 以 `MASS_{bin}` 前缀注入
- 训练参数：`batch_size=4096` tokens，`accum_count=4`，`learning_rate=2.0`，`warmup_steps=8000`
- `train_steps=50000`，`valid_steps=2000`，`save_checkpoint_steps=2000`
- 序列长度：`max_src_len=500`，`max_tgt_len=500`
- 推理：`beam_size=20`，`n_best=20`，`topk=5`
- 数据与产物：`data/processed/v6b/`，`results/checkpoints/v6b/`，`logs/v6b/`

### Checkpoint 落盘格式（当前脚本）

以下与 **`Trainer` 断点恢复、`load_best_model_at_end`、权重加载** 对齐；实现见各脚本中的 **`V5Trainer._save`**。

- **适用脚本**：`train_v5a.py`、`train_v5b.py`、`train.py`、`train_v5b1.py`（LoRA 内核）、`train_v5a2.py`、`train_v5b2.py`（LoRA 入口，仍由 `train_v5b1` 内核落盘）。
- **每个 `checkpoint-*`（及 Trainer 写入的中间存档目录）通常包含**：
  - **`pytorch_model.bin`**：完整 **`V5SCLM.state_dict()`**（底座 T5 + conditioning 相关权重），供 HF **`Trainer`** 按官方路径加载；
  - **`v5_state.pt`**：`reaction_embed`、`cfg_*`、`token_to_idx` 等字段的冗余快照（便于核对或与推理侧约定对齐）；
  - **tokenizer 目录文件**（随 checkpoint 一并写入）；
  - **`training_args.bin`**；
  - **`trainer_state.json`**：是否在同级目录取决于 transformers 版本与保存布局（以磁盘为准）。
- **主线（`train_v5a` / `train_v5b`）收尾**：训练结束后仍会 **`model.save`** → **`{output_dir}/best/`**，用于 **`V5SCLM.load`** 的推理 bundle。
- **LoRA 完整训练（`train_v5b1`）收尾**：若 **`model.base_model`** 为 **`PeftModel`**，在写入 **`best/`** 前会先 **`merge_and_unload()`**，再 **`model.save`**，避免仅保留 adapter 导致 **`V5SCLM.load`** 不完整。
- **历史实验说明**：若某次旧目录里**没有** **`pytorch_model.bin`**（仅有早期 **`save_pretrained`/adapter 形态**），则 **`load_best`/resume** 行为与**现行脚本**不一致；对比不同批次结果时请先看目录内容再解读指标。

---

## 基准测试计划（V5，1-5 全量执行）

目标：在训练结束后快速得到“效果-稳健性-成本”三维结论，支持 V5A/V5B 选型与参数冻结。

### 1) 主评估集（必须）

统一使用 `src/eval/metrics.py`，固定输出：

- Validity
- Exact Match（InChIKey-14）
- PPM pass rate（分母固定 `target_mass`）
- Tanimoto

每次评估必须附带：模型分支、checkpoint、推理参数、数据版本标识（时间戳或 hash）。

### 2) 对比实验层（核心）

最小可比矩阵（先跑单 seed 出初版）：

- Backbone：V5A（MolT5）vs V5B（ReactionT5）
- CFG：off / on
- TTA：off / on（固定一个档位）

即 `2 x 2 x 2 = 8` 组。  
初版完成后，对最优 2-3 组补 3 seeds，汇报均值与方差。

### 3) 可靠性层（稳健性）

对最优候选组做分层评估：

- OOD scaffold 子集
- 长/复杂 SMILES 子集
- PPM 阈值边界子集

输出子集指标与失败类型分布（invalid / mass mismatch / wrong structure）。

### 4) 工程效率层（落地）

统一记录：

- 吞吐（samples/s）
- 单样本延迟
- 显存峰值
- CFG/TTA 相对额外开销

最终形成“效果-成本”对照表，支持部署决策。

### 5) 外部基线层（BioTransformer + EPA CTS）

目标：将 SC-CLM 与行业常用规则/知识库工具做同口径对照，给出“是否优于外部基线”的结论。

**正式基线（2026-05-24）**：R **patRoon** 导出 + `scratch/eval_patoon_external.py`（Setting-2 PPM 重排）。主表：`results/benchmark/v5/summary/summary_external_patoon_comparison.csv`、`summary_external_patoon_by_token.csv`。进度与结论：**`results/benchmark/v5/BENCHMARK_PROGRESS.md`** §0、§7。

纳入基线：

- **patRoon BioTransformer**（allHuman / superbio / cyp450 / hgut / env / ecbased + narrow/broad）
- **patRoon CTS**（分库 + union；**CTS 环境转化已由 patRoon BT 流程覆盖**，与 BT 一并评估，**不再单独跑 EPA CTS API**）
- **patRoon library**
- 历史探索：Python JAR `benchmark_bt.py`（`summary_external.csv`，不作主结论）

公平性约束（强制）：

- 同一测试集（`data/processed/test.csv`）
- 同一候选条数口径（建议统一 top-k）
- 同一后处理（SMILES 合法性、去重、无效项过滤）
- 同一评估脚本（`src/eval/metrics.py`）
- 同一 PPM 口径（分母固定 `target_mass`）

输出要求：

- 主表增加外部基线列：`V5A / V5B / BioTransformer / EPA CTS`
- 同时报告主指标与工程开销（吞吐、延迟）
- 对失败样本做类型归因（invalid / mass mismatch / structure mismatch）

### 执行顺序（固定）

1. 先完成 8 组最小矩阵（阶段A：主结论优先）
2. 对最优组补 seeds 与统计稳定性
3. 对最优组补 3)+4)（阶段B：稳健性与工程落地）
4. 运行 5) 外部基线层（patRoon BT + CTS，Setting-2）并统一汇总（**EPA CTS API 不纳入**）
5. 汇总到统一目录（建议 `results/benchmark/v5/`）

---

## 数据与评估口径

### 数据流程

`data/raw/*.csv -> src/data/preprocess.py -> data/processed/{train,val,test}.csv`

V6 数据流程：
`data/raw/*.csv + results/cts_augmented/cts_all_likely_*.csv -> src/model/v6/prepare_data.py -> data/processed/v6_{train,val,test}.csv`

V6b 额外输出：
`data/processed/v6_{train,val,test}.csv -> src/model/v6b/prepare_data.py -> data/processed/v6b/{src,tgt}-{train,val,test}.txt`

当前规模：
- `v6_train.csv`: 8,361 rows（raw 8,156 + CTS 3,948，去重后）
- `v6_val.csv`: 412 rows
- `v6_test.csv`: 152 rows

约束：

- 使用 RDKit 做 SMILES 合法性检查
- 使用 InChIKey-14 做去重与精确匹配
- Murcko scaffold 划分，确保 train-test OOD

### 评估标准（唯一口径）

统一使用 `src/eval/metrics.py`：

- Validity: `Chem.MolFromSmiles(pred) is not None`
- Exact Match: InChIKey-14(pred) == InChIKey-14(target)
- PPM fidelity: `abs(delta_m_pred - delta_m_theoretical) / target_mass * 1e6 <= threshold`
- Tanimoto: Morgan 指纹相似度

> PPM 分母必须是 `target_mass`，不是 `pred_mass`。

---

## 规则来源（精简版）

项目规则以 `rules.md` 为统一入口，细则来源于：

- `.antigravity-rules/coding-standards.md`
- `.antigravity-rules/chemistry-domain-rules.md`
- `.antigravity-rules/ml-dev-style.md`

若 `rules.md` 与旧文档不一致，以 `rules.md` 为准。

---

## Project Memory（强制执行）

统一使用 `project_memory.md` 记录关键经验，执行规则如下：

1. **踩坑即时记录**：出现报错、回滚、评估异常、环境坑时，立即写入 memory（不延后）。
2. **任务完成后沉淀**：每个完整任务结束，必须追加总结：
  - 项目内容与结论
  - 最佳实践
  - 关键约束
  - 已确认的坑与规避方式
3. **记录优先级**：先记录再继续下一轮大改，确保上下文不丢失。

---

## 会话交接（Handoff，强制执行）

在以下任一时刻，必须更新 `./handoff.md`：

- 准备离开/关闭 CC 窗口；
- 或会话上下文使用量接近 70%。

`handoff.md` 固定结构如下（不得删项）：

```markdown
## 当前在做什么
## 已经试过的方案和结果(含失败的)
## 下一步计划(3-5条 actionable)
## 关键文件路径(相对路径，一行一个)
## 还没搞清楚的问题
```

要求：保证 3 天后打开该文件，30 秒内可以继续工作。

---

## 日常回复格式（非强制）

平时对话不强制使用固定 5 段模板，可按问题复杂度自然回复。  
仅在“离开/关闭窗口”或“context 接近 70%”时，强制更新 `handoff.md`。

---

## Skills 使用策略（按需取用）

- 本项目优先使用与 SC-CLM 直接相关的 `skills/` 与 `.claude/agents/`。
- `.agents/skills/` 下通用论文写作类 skills 默认不参与常规上下文（见 `.cursorignore`）。
- 仅在用户明确要求相关能力时，再临时读取对应 skill。

---

## 已知技术债


| #   | 问题                                       | 优先级 | 说明                          |
| --- | ---------------------------------------- | --- | --------------------------- |
| 1   | V4 旧脚本 PPM 分母历史不一致                       | 中   | 仅影响历史对比，不影响 V5 口径           |
| 2   | `src/evaluation/` 与 `src/eval/` 命名并存     | 低   | 后续统一                        |
| 3   | `src/routing/optimize_routing.py` 主入口空实现 | 低   | 实验路径                        |
| 4   | 长训练日志含大量 RDKit parse warning             | 中   | 需结合 `trainer_state.json` 解读 |
| 5   | opennmt-py (V6b) 与 transformers (V6a) torch 版本冲突 | 高   | 需独立 conda 环境运行 V6b 训练 |


---

## 改动前洁癖审查（必做）

对任何代码改动，先切换到“代码洁癖工程师”视角做一轮自检：

1. 是否 over-engineering（过度抽象、过度封装、过度参数化）；
2. 是否存在更短、更直接、更可维护的实现；
3. 若发现 over-engineering，**直接简化**，默认优先可读性与稳定性。

---

## 维护检查清单

```bash
# 数据完整性
python scratch/fix_and_finalize.py

# PPM 分母检查
rg "pred_mass|target_mass" src/model/v5 src/eval/metrics.py

# 三分集规模快速检查
python -c "import pandas as pd; [print(s, len(pd.read_csv(f'data/processed/{s}.csv'))) for s in ['train','val','test']]"
```

