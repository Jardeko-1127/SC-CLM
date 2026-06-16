# SC-CLM 模型说明（V5A / V5B）

## 1. 任务定义

SC-CLM 目标是学习环境转化反应中的 `parent SMILES -> product SMILES` 映射，结合反应类型和质量差约束，预测候选转化产物。

---

## 2. 当前模型架构

### V5 公共骨架

- 编码器-解码器框架（T5 family）
- 反应条件采用 **embedding-conditioned** 注入，而非早期 token sandwich
- 训练使用 CFG dropout，推理使用 guidance 进行条件强化

### V5A（主线）

- Backbone: `laituan245/molt5-small`（约 77M）
- 训练脚本: `src/model/v5/train_v5a.py`
- 日志: `logs/v5a/train.log`
- 检查点: `results/checkpoints/v5a/`

### V5B（实验线）

- Backbone: `sagawa/ReactionT5v2-forward`
- 训练脚本: `src/model/v5/train_v5b.py`
- 日志: `logs/v5b/train.log`
- 检查点: `results/checkpoints/v5b/`

### V5A1 / V5B1（LoRA 校准线）

- 目的：在低显存条件下快速验证训练稳定性与吞吐，不替代 V5A/V5B 正式主线结果
- LoRA 配置（统一）：`r=16`，`alpha=32`，仅训练 attention 层（`q,k,v,o`）
- 低显存口径：`per_device_train_batch_size=1`，`gradient_accumulation_steps=4~8`（默认 8）
- 序列长度：
  - MolT5（V5A1）=`max_input_len=128, max_output_len=128`
  - ReactionT5（V5B1）=`max_input_len=256, max_output_len=128`
- 训练脚本：
  - `src/model/v5/train_v5a1.py`（MolT5 + LoRA）
  - `src/model/v5/train_v5b1.py`（ReactionT5 + LoRA）
- 日志与检查点：
  - V5A1：`logs/v5a1/train.log`，`results/checkpoints/v5a1_calib/`
  - V5B1：`logs/v5b1/train.log`，`results/checkpoints/v5b1_calib/`

---

## 3. 数据与划分

- 原始数据来源：`data/raw/`
- 处理输出：`data/processed/train.csv`, `val.csv`, `test.csv`
- 关键规则：
  - RDKit 有效性校验
  - InChIKey-14 去重
  - Murcko scaffold 分层划分（避免 train-test scaffold 泄露）
  - 10 类转化 token（见 `src/data/preprocess.py`）

---

## 4. 评估指标（canonical）

统一以 `src/eval/metrics.py` 为准：

- **Validity**: `MolFromSmiles(pred) != None`
- **Exact Match**: InChIKey-14 精确匹配
- **PPM fidelity**: `abs(delta_m_pred - delta_m_theory) / target_mass * 1e6 <= threshold`
- **Tanimoto**: Morgan 指纹相似度

注意：PPM 分母必须使用 `target_mass`。

---

## 5. 当前训练读取方式

建议用两层信息源：

1. `logs/v5a/train.log`：检查点保存节奏与运行状态  
2. `results/checkpoints/v5a/checkpoint-*/trainer_state.json`：最新 `global_step`、`epoch`、`eval_*` 指标和 `best_metric`

---

## 6. 已知现象

- 长跑评估阶段会出现大量 RDKit SMILES Parse Warning（日志噪声较大）
- 在当前环境中可能出现伴随双进程（`venv` + `Python312`）的启动形态，需要以 checkpoint 持续增长来判断训练是否健康前进
