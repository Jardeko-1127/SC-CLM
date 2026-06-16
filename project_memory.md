# Project Memory（SC-CLM）

> 用于沉淀“踩坑、约束、最佳实践”。  
> 规则：踩坑立即记；每个任务完成必须追加总结。

---

## 记录模板

### [YYYY-MM-DD HH:MM] 事件标题
- 背景：
- 现象/报错：
- 根因：
- 处理动作：
- 最佳实践：
- 关键约束：
- 后续提醒：

---

## 2026-05-08 10:18 文档与协作规范固化
- 背景：项目进入 V5A/V5B 双线，文档与协作规则需要统一。
- 现象/报错：历史文档分散，规则入口多，skills 容易引入无关上下文。
- 根因：缺少单一 memory 入口与强制沉淀流程。
- 处理动作：
  - `CLAUDE.md` 新增 Project Memory 强制流程与“改动前洁癖审查”；
  - `rules.md` 增加 memory 与 over-engineering 简化规则；
  - 新增本文件 `project_memory.md` 作为统一记录入口。
- 最佳实践：
  - 先记录再迭代，避免重复踩坑；
  - 任务结束必须沉淀可复用经验。
- 关键约束：
  - 全中文协作；
  - PPM 分母使用 `target_mass`；
  - 训练健康度以 `trainer_state.json` 和 checkpoint 连续增长为准。
- 后续提醒：后续每次训练异常、评估异常、环境异常都按模板即时追加。

## 2026-05-08 13:42 Claude-mem Hook 在 Windows 下报错修复
- 背景：安装 `claude-mem` 后，Claude CLI 在 `Stop` 阶段持续出现 non-blocking hook 报错。
- 现象/报错：
  - `/usr/bin/bash: line 1: printf: write error: Permission denied`
  - `Error: Bun not found. Please install Bun: https://bun.sh`
- 根因：
  - hook 仍走 bash 管道脚本，在 Windows + PowerShell 场景下易触发 stdout 写入异常；
  - Hook 子进程环境未稳定继承 Bun 路径，导致 `bun-runner` 认为 Bun 不存在。
- 处理动作：
  - 新增 `C:\\Users\\jarde\\.claude\\scripts\\claude-mem-hook.ps1` 作为统一 hook 包装器；
  - hooks 改为调用 PowerShell 包装器（至少覆盖 `Stop -> summarize`），不再依赖 bash/printf；
  - 包装器显式注入 Bun 路径：`%LOCALAPPDATA%\\Microsoft\\WinGet\\Links` 与 `%USERPROFILE%\\.bun\\bin`。
- 最佳实践：
  - Windows 下 third-party hook 默认优先 PowerShell，不要混用 bash 管道构造；
  - 对 runtime 依赖（node/bun）在 hook 内显式校验和补 PATH。
- 关键约束：
  - 修改了插件缓存路径下 hooks，插件 update 后可能被覆盖；
  - 覆盖后需重新应用本修复。
- 后续提醒：若再次出现同类报错，先检查 hooks 是否被更新覆盖，再检查 `claude-mem-hook.ps1` 是否还在。

## 2026-05-08 15:46 V5 基准测试范围与分阶段执行确认
- 背景：训练进行中，需要提前冻结 benchmark 口径，确保收敛后可直接出结论。
- 现象/报错：无运行报错，本次为方案决策与文档固化。
- 根因：若不提前统一口径，训练完成后容易出现“指标齐全但不可比”或“缺工程成本维度”。
- 处理动作：
  - 确认基准测试 1-4 全部执行（主评估、对比实验、可靠性、工程效率）；
  - 确认先阶段A（1+2）后阶段B（3+4）的执行顺序；
  - 在 `CLAUDE.md` 新增“基准测试计划（V5，1-4 全量执行）”章节并固定输出口径。
- 最佳实践：
  - 先跑最小可比矩阵（2x2x2）拿主结论，再对最优组做稳健性和成本评估；
  - 所有结果必须附参数元信息，避免后续无法复盘。
- 关键约束：
  - PPM 分母严格使用 `target_mass`；
  - 对比实验必须使用同一 test 集版本与统一评估脚本。
- 后续提醒：训练结束后按 `CLAUDE.md` 既定顺序执行，不临时改指标定义与目录结构。

## 2026-05-08 15:50 外部基线（BioTransformer / EPA CTS）纳入确认
- 背景：内部对比（V5A/V5B + CFG/TTA）可回答“谁更优”，但无法回答“是否优于外部工具”。
- 现象/报错：无运行报错，本次为评估框架扩展决策。
- 根因：缺少外部基线会导致实验结论仅具内部参考价值，论文/汇报说服力不足。
- 处理动作：
  - 将基准测试由“1-4 全量”扩展为“1-5 全量”；
  - 新增第5层：BioTransformer 与 EPA CTS 的同口径对比；
  - 在 `CLAUDE.md` 增加外部基线公平性约束与输出要求。
- 最佳实践：
  - 外部工具输出必须先做统一后处理，再走同一 `metrics.py`；
  - 保持同一 test 集与同一 top-k 口径，避免比较偏差。
- 关键约束：
  - 仍使用 `target_mass` 作为 PPM 分母；
  - 外部基线比较必须与主模型共享相同评估脚本与过滤规则。
- 后续提醒：训练结束后先完成阶段A/B，再补外部基线并统一汇总到 `results/benchmark/v5/`。

## 2026-05-08 15:59 外部基线比较任务重定义（A/B 双任务 + 三种信息设置）
- 背景：SC-CLM 输入 `SMILES + Δmz`，而 BioTransformer/CTS 输入 `SMILES` 并生成候选集，直接单口径比较存在方法学偏差。
- 现象/报错：无运行报错，本次为比较框架公平性修订。
- 根因：两类方法的信息条件不同，若不分任务与信息设置，结论容易失真。
- 处理动作：
  - 在 `benchmark_plan.md` 新增“任务定义与公平性协议”；
  - 固定双任务：任务A（目标导向预测）+ 任务B（候选生成覆盖）；
  - 固定三种信息设置：Setting-1 原生输入、Setting-2 统一质量重排、Setting-3 去质量信息下限；
  - 新增按任务的数据建议与任务B补充指标（Hit@k/Recall@k 等）。
- 最佳实践：
  - 外部基线建议以 Setting-2 作为主结论，Setting-1 作为现实场景对照；
  - 任务B必须报告候选覆盖能力，不能只报精确命中。
- 关键约束：
  - 所有 setting 仍使用同一后处理与 `metrics.py`；
  - PPM 口径继续固定 `target_mass` 分母。
- 后续提醒：最终汇报时明确区分”任务A结论”和”任务B结论”，避免跨任务混比。

## 2026-05-08 21:13 V5A 训练过程验证（非 benchmark）
- 背景：用户要求先保留训练结果，优先做训练过程验证，不启动 benchmark。
- 现象/报错：训练任务 shell 结束码曾显示 `exit_code=1`，需要确认是否真实失败。
- 根因：任务退出码异常不等于训练失败，需以 `trainer_state.json` 与 checkpoint 完整性为准。
- 处理动作：
  - 校验 checkpoint 完整性：`checkpoint-49305` 与 `checkpoint-51900` 均存在且包含 `trainer_state.json`、`config.json`；
  - 校验训练收敛轨迹：`epoch=60.0`、`global_step=51900`、`eval_count=60`，eval 步长固定 865；
  - 校验稳定性：训练 loss 记录 1038 条，无 NaN/Inf；
  - 校验最佳点：`best_metric=19.3`，最佳 checkpoint 为 `checkpoint-49305`（epoch 57）。
- 最佳实践：
  - 训练健康度优先看 `trainer_state.json` 的完成度、最佳点和 eval 连续性；
  - shell 退出码可作告警，但不能单独作为“训练失败”判据。
- 关键约束：
  - 当前结论仅针对“训练过程验证”，尚未进入 benchmark 与外部基线对比。
- 后续提醒：后续默认使用 `checkpoint-49305` 作为 V5A 主评估权重。

## 2026-05-08 21:18 V5B 启动前连通性阻塞（ReactionT5 基座不可达）
- 背景：V5A 完成后准备切换启动 V5B 训练。
- 现象/报错：`from_pretrained('sagawa/ReactionT5v2-forward')` 触发 `httpx.ConnectTimeout [WinError 10060]`。
- 根因：当前机器无法稳定访问 Hugging Face；且本地无 `ReactionT5v2-forward` 快照缓存。
- 处理动作：
  - 已创建 `logs/v5b/` 与 `results/checkpoints/v5b/`；
  - 已确认 `C:\\Users\\jarde\\.cache\\huggingface\\hub` 下无该模型缓存，暂无法离线启动。
- 最佳实践：
  - V5B 开跑前先做模型可达性 smoke test；
  - 若网络不稳，先在可联网机器预拉取快照，再拷贝到本机走本地路径。
- 关键约束：
  - 在未获取模型快照前，不建议直接启动 `train_v5b.py`（会在加载阶段超时失败）。
- 后续提醒：优先准备本地快照或改到可联网服务器执行 V5B。

## 2026-05-08 17:30 V5A 训练完成（epoch 60/60）

- 背景：V5A（MolT5-Small + embedding-conditioned + CFG）完整 60 epoch 训练结束。
- 训练概况：
  - 启动：2026-05-07 19:22（最终一次成功启动）
  - 结束：2026-05-08 17:26，**总耗时 ~22h04m**
  - 配置：bf16，effective batch=32，CFG dropout=0.15，augment_n=5
  - 训练集 5530 base × 5 aug = 27650 effective；验证集 399
  - 总步数 51,900（60 epoch）
- **最终最佳指标（checkpoint-49305，epoch 57）**：
  - `eval_ppm_pass_rate` = **19.3%**（best）
  - `eval_loss` = **0.54893**（best）
  - `eval_exact_match` = 5.01%
  - `eval_validity` = 70.18%
- **末点指标（checkpoint-51900，epoch 60）**：
  - `eval_ppm_pass_rate` = 17.04%
  - `eval_loss` = 0.55150
  - `eval_exact_match` = 4.76%
  - `eval_validity` = 70.68%
- **关键发现**：
  - **明显过拟合**：最佳点在 epoch 57（49305 step），之后 ppm_pass_rate 从 19.3% 跌至 17.04%；
  - eval_loss 在 epoch 57 后也出现反弹（0.54893 → 0.55150），训练应提前终止；
  - Validty 在后期稳定在 ~70%，但 PPM 与 EM 随过拟合退化。
- 训练曲线总结（PPM pass rate 演进）：
  - Step 865-6920（epoch 1-8）：0~0.25%（冷启动）
  - Step 7785-17300（epoch 9-20）：0.25%~5.51%（缓慢提升）
  - Step 18165-32870（epoch 21-38）：5%~15%（稳步上升）
  - Step 33735-49305（epoch 39-57）：15%~19.3%（持续爬升至峰值）
  - Step 50170-51900（epoch 58-60）：18.55%→17.04%（过拟合退化）
- 处理动作：
  - `trainer_state.json` 中 `best_metric=19.3`，`best_model_checkpoint=checkpoint-49305`；
  - 后续推理/评估以 **checkpoint-49305** 为准，不使用末点 checkpoint-51900。
- 最佳实践：
  - 长时间训练必须依赖 `trainer_state.json` 的 `best_model_checkpoint`，不盲目用末点；
  - 过拟合信号（eval_loss 反弹 + 核心指标回落）在 epoch 57 已出现，实际应启用 early stopping；
  - 训练监控应同时关注 eval_loss 与 eval_ppm_pass_rate 两条曲线。
- 关键约束：
  - V5A 最佳 checkpoint 固定为 `results/checkpoints/v5a/checkpoint-49305`；
  - 后续所有 V5A 推理与评估必须明确注明 checkpoint 版本（49305 vs 51900）；
  - PPM 口径继续固定 `target_mass` 分母。
- 后续提醒：
  - 基准测试层面：按 CLAUDE.md 计划，使用 checkpoint-49305 执行 5 层 benchmark；
  - 训练层面：V5B 启动前考虑加入 early stopping（patience=5 epoch），避免再次跑满后过拟合；
  - 模型层面：可尝试加大 CFG dropout 缓解过拟合，或降低 lr 做更长平缓收敛。

## 2026-05-09 V5A1/V5B1：`train_v5b1.py` 默认完整 LoRA + `V5B1_MAX_STEPS` / `V5B1_NUM_EPOCHS`

- 项目内容与结论：
  - **`V5B1_MAX_STEPS` 默认 `-1`**：`Seq2SeqTrainingArguments.max_steps=-1`，按 **`V5Config.num_epochs`（默认 60）** 跑 **完整 LoRA**，不再默认 300-step。
  - **`V5B1_MAX_STEPS` 为正整数**：进入 **短跑校准**（无按 epoch eval/save），用于冒烟、吞吐或快速诊断。
  - **`V5B1_NUM_EPOCHS`**：若 shell 中设置（非空字符串），覆盖内核里的 `num_epochs`，便于本机缩短试验轮数。
  - **完整 LoRA** 时对齐主线评估习惯：每 epoch eval/save、`predict_with_generate`、`load_best_model_at_end`、`ppm_pass_rate` 选优、早停 patience=15、`save_total_limit=3`，并在 **`{output_dir}/best/`** 另存一份最佳快照。
  - **`train_v5a1.py`** 不再 `setdefault(V5B1_MAX_STEPS, 300)`，与内核默认「完整 LoRA」一致。
- 最佳实践：
  - 新一轮长跑务必 **`V5B1_OUTPUT_DIR` / `V5B1_LOG_FILE` 时间戳或独立子目录**，避免覆盖历史 checkpoint/日志。
  - **`logging_steps=50`** 时，HF 要等 **50 个优化器 step** 才打 loss；配合 **`grad_accum=8`**，慢速 CPU/GPU 可能在很长时间内只有 `Starting…`，易被误判为卡死。
  - `train_v5b1.py` 已加 **`Heartbeat` 回调**（前 15 个 optimizer step 必打日志）与完整 LoRA **`logging_first_step=True`**。
  - 需要 300-step 校准时 **显式** `$env:V5B1_MAX_STEPS='300'`（或按需 20），不要在文档中假设仍为默认。
- 关键约束：
  - MolT5（V5A1）仍建议 **`V5B1_FORCE_FP16=0`（默认）**；ReactionT5（V5B1）默认 fp16；与「完整 LoRA」正交。
  - 正式 benchmark 主表仍以 **V5A/V5B** 为准；A1/B1 为低显存 LoRA 线（现为默认完整训练 + 可选短跑）。
- 已确认的坑与规避方式：
  - 嵌套 PowerShell `-Command` 易导致 `$_.CommandLine` 被吃掉，进程核查请在**单层** PowerShell 执行 `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'train_v5a1|train_v5b1' }`，或直接 tail 独立日志文件。

## 2026-05-09 14:xx V5B1 低显存 LoRA 训练口径改造（MolT5/ReactionT5 通用）

- 项目内容与结论：
  - 将 `src/model/v5/train_v5b1.py` 从“仅 ReactionT5 校准”改为“MolT5/ReactionT5 通用低显存口径”（后续已演进为 **默认完整 LoRA**，见上条 memory）；
  - 固化低显存策略：**ReactionT5：`FP16` + gradient checkpointing + LoRA(attention-only)**；MolT5 路径 fp32 优先（见 V5A1 条目）；
  - LoRA 参数固定：`r=16`、`alpha=32`、`target_modules=["q","k","v","o"]`；
  - batch 口径固定：`per_device_train_batch_size=1`，`gradient_accumulation_steps` 限制在 `4~8`。
- 最佳实践：
  - 8GB 显存场景优先以 LoRA + checkpointing 作为主训练模式，避免全参数微调导致的 OOM/卡顿；
  - 保持 `micro_batch=1`，通过累积步数调节等效 batch（4~8）更稳；
  - 序列长度按 backbone 家族区分：MolT5 `128/128`，ReactionT5 `256/128`。
- 关键约束：
  - 脚本内强校验：`micro_batch` 必须为 1；`grad_accum` 必须在 4~8；checkpointing 必须开启；
  - backbone 仍可通过环境变量切换，但默认保持 ReactionT5 路径不变；
  - `reaction_embed_dim` 改为读取 backbone `AutoConfig.d_model` 自动对齐，避免维度不匹配。
- 已确认的坑与规避方式：
  - 仅调 batch/累积并不能本质提速，主提速仍来自序列长度控制；
  - LoRA 需要 `peft` 可用（当前 venv 已确认安装）；
  - 在 T5 上做 LoRA 时目标模块应限制在 attention 投影层，避免不必要的可训练参数扩张。

## 2026-05-09 14:xx V5A1 封装（MolT5-small + LoRA）

- 项目内容与结论：
  - 新增 `src/model/v5/train_v5a1.py`，作为 `MolT5-small + LoRA` 的轻量包装脚本；
  - 复用 `train_v5b1.py` 的通用 LoRA 训练核心，避免复制大段训练逻辑；
  - 默认输出与日志分离：`results/checkpoints/v5a1_calib`、`logs/v5a1/train.log`。
- 最佳实践：
  - 对 A/B 两条线共用一套 LoRA 训练内核时，用 wrapper 仅改默认环境变量最稳；
  - 保持 `micro_batch=1`、`grad_accum=8` 作为 8GB 显存默认档位，减少环境差异。
- 关键约束：
  - `train_v5b1.py` 已支持 `V5B1_OUTPUT_DIR`/`V5B1_LOG_FILE` 覆盖，wrapper 必须设置以免产物混写；
  - MolT5 默认模型路径使用已验证可加载的本地快照路径。
- 已确认的坑与规避方式：
  - 不能在 `Seq2SeqTrainingArguments` 里开启 wrapper 模型的 gradient_checkpointing（会触发缺少方法报错）；
  - 正确做法是在 `base_model` 手动启用 checkpointing，并将 training args 的该开关关闭。

## 2026-05-09 18:xx V5B1/V5A1 LoRA 校准结论（阶段性）

- 项目内容与结论：
  - `V5B1 (ReactionT5 + LoRA)` 300-step 已稳定完成，速度和 loss 曲线正常；
  - `V5A1 (MolT5 + LoRA)` 初版出现 `loss=0 / grad_norm=nan / lr=0` 异常，后续通过稳定性修复恢复正常；
  - 稳定性修复核心：MolT5 默认关闭 fp16（保留 LoRA + grad checkpoint），并增加前向 sanity 检查日志。
- 最佳实践：
  - LoRA 校准阶段先看 “可训练参数比例 + 每步日志 + 前向 sanity loss”，再决定是否放大到全量；
  - Windows 下长任务建议避免依赖会话托管终端，必要时使用脱离会话启动以减少异常中断。
- 关键约束：
  - `V5B1` 保持 ReactionT5 口径：`max_in/max_out=256/128`，`micro_batch=1`，`grad_accum=8`；
  - `V5A1` 保持 MolT5 口径：`max_in/max_out=128/128`，`micro_batch=1`，`grad_accum=8`，默认 `fp16=False`；
  - 两条线 LoRA 统一：`r=16`，`alpha=32`，`target=q,k,v,o`。
- 已确认的坑与规避方式：
  - 会话级中断会导致训练进程 `4294967295` 退出，需与参数/代码异常区分；
  - `V5A1` 包装脚本需强制覆盖 family，避免继承上一次 shell 环境变量导致误用 ReactionT5；
  - 共用内核日志会标明 **`full LoRA`** 或 **`calibration`**（由 `V5B1_MAX_STEPS` 正负决定）；不以文案字样判断 backbone，应以 **`Model family`** 字段为准。

## 2026-05-09 18:xx 训练运行分工与保存策略确认

- 项目内容与结论：
  - 用户确认运行分工：**本机跑 A1/B1，服务器跑 A/B**；
  - 用户要求所有相关结果都保留，不删除、不覆盖。
- 最佳实践：
  - A1/B1 作为低显存 LoRA 线：**默认完整 LoRA**，亦可显式短跑校准；本机持续迭代；
  - A/B 作为正式主线，优先放到服务器执行，减少本机会话中断风险。
- 关键约束：
  - 新增运行使用独立目录或时间戳命名；
  - benchmark 主表默认仅纳入 A/B（A1/B1 仅做校准对照）。
- 已确认的坑与规避方式：
  - 若复用旧目录易导致结果串写，必须避免覆盖写入；
  - 会话终止可能中断本机训练，正式长跑应优先服务器执行。

## 2026-05-09 18:xx V5A1 300-step 干净复跑最终判定

- 项目内容与结论：
  - 在独立输出目录 `results/checkpoints/v5a1_calib_20260509_1832` 与独立日志 `logs/v5a1/train_20260509_1832.log` 下完成单次 300-step；
  - 训练 `exit_code=0`，无中断；
  - 关键数值恢复正常并可用：`loss` 随步数明显下降（约 `851.8 -> 64.41`），`grad_norm` 为有限值，`learning_rate` 按调度衰减。
- 最佳实践：
  - A1/B1 最终判定必须使用“单次干净日志 + 独立输出目录”结果，避免多轮串写干扰；
  - MolT5 LoRA 默认 `fp16=False` 的稳定性优于此前 fp16 路径。
- 关键约束：
  - 后续 A1 训练与对比应优先引用本次干净复跑结果；
  - 保持本机 A1/B1 与服务器 A/B 的运行分工不变。
- 已确认的坑与规避方式：
  - 历史 `logs/v5a1/train.log` 存在多轮混写，不适合最终判定；
  - 终端托管任务偶发退出可通过独立复跑与独立日志进行隔离验证。

## 2026-05-10 `_status.txt` 与 TrainerState 同步

- `src/model/v5/train_v5b1.py` 内 `_RepoRootStatusCallback`：rank0 在 `on_log` / `on_evaluate` / `on_save` / `on_train_end` 原子写入仓库根 `_status.txt`（默认），环境变量 **`V5_STATUS_FILE`** 可改路径；字段含 `Last step`、`epoch`、`best_model_checkpoint`、`best_metric`、累计 `Eval:` 行；若当前步对应 checkpoint 已落盘则附加 **`disk_trainer_state.json global_step`** 校验。

## 2026-05-11 LoRA 断点续训

- **`V5B1_RESUME_FROM_CHECKPOINT`**：`auto` / `1` / `true` → 在 **`V5B1_OUTPUT_DIR`** 下选步数最大的 **`checkpoint-*`**（需含 `trainer_state.json` 与 `pytorch_model.bin`）；否则为相对 output_dir 或绝对路径。`train_v5b1.py` 内 **`trainer.train(resume_from_checkpoint=...)`**。

## 2026-05-11 系统 Python 与 venv 双开 LoRA

- **成因**：Cursor/IDE 用 **`Python312`** 跑 `train_v5a1.py` 时，`sys.executable` 传给 `train_v5b1` 子进程仍是系统 Python，与 **`venv\Scripts\python.exe`** 各跑一套、抢 GPU/OUTPUT_DIR。
- **代码**：`train_v5a1.py` 在调用内核前若检测到仓库 **`venv\Scripts\python.exe`** 存在且当前解释器不是该路径，则 **`subprocess` 用 venv 重跑自身**；`train_v5b1.py` 在 **导入 torch 之前** 对 **`__main__`** 做同样切换。
- **运维**：**`scratch/kill_nonvenv_lora_train.ps1`** ——结束命令行含 **`train_v5a1`/`train_v5b1`** 且解释器路径 **不含** **`venv\Scripts\python.exe`** 的进程，保留 venv 那一套。

## 2026-05-20 V5 基准测试进度文档

- **背景**：阶段 A/B 与 BT 基线已跑完，需单一入口记录产物与口径。
- **处理动作**：新增 **`results/benchmark/v5/BENCHMARK_PROGRESS.md`**（指标表、路径、BT 模式说明、待办）；在 **`benchmark_plan.md`** 文首增加指向该文件的链接。
- **最佳实践**：后续每轮 benchmark 更新 `BENCHMARK_PROGRESS.md` 的「最后更新」与待办，避免与 `CLAUDE.md` 表格重复维护两处口径。
- **待办**：EPA CTS（`run_v5_benchmark_external.py --only cts`）；BT 可选改进 `TOKEN_MODE_MAP`（如 `[TRANS_OXIDATION]`）后重跑需新 `run_id`。

## 2026-05-24 patRoon 外部基线全量评估与文档固化

- **背景**：需将 R patRoon 导出的 BT 多途径 / CTS / library 与 SC-CLM（V5A/B/A1/B1）在同口径下对比，并支持按 `token` 分层与候选池指标。
- **项目内容与结论**：
  - 评估脚本 **`scratch/eval_patoon_external.py`**：Setting-2（候选池 PPM 升序 top-1）、`V4Metrics`、PPM 分母 `target_mass`。
  - **全量逐项**：`summary_external_patoon_comparison.csv`（superbio PPM **83.49%** > V5B **72.48%**；V5B1 **Exact 30.28%** 仍最强）。
  - **按 token**：`summary_external_patoon_by_token.csv`（69 行，含 V5A/B/A1/B1）；OXIDATION/DEMETHYLATION 上 narrow PPM 明显优于 V5B；GLUCURONIDATION 等 V5B 更优。
  - 旧 Python JAR BT（`summary_external.csv`，PPM 31.19%）仅作探索性，**不作主结论**。
- **最佳实践**：
  - 外部主表用 **narrow + broad** 或单途径 superfio/allHuman，勿无脑全途径 union。
  - `recall@5` 应在 **PPM 排序后** 的前 5 候选上计算；报告同时给出 `mean_n_candidates`。
  - 仅更新分层表：`python scratch/eval_patoon_external.py --by-token-only`（依赖已有 `predictions_*.csv`）。
- **关键约束**：
  - 训练日志 `exact_match` = InChIKey-14；benchmark CSV 中 `exact_match_pct` = 规范 SMILES 完全匹配，不可混读。
  - library 约半数样本无命中，Validity/Tanimoto 与 Exact 需分开解读。
- **已确认的坑与规避方式**：
  - Python `benchmark_bt.py` 大量样本仅 `env` 模式 → PPM 偏低，不可与 patRoon 对比混用。
  - `TOKEN_TO_BT_PATHWAYS` 在 `eval_patoon_external.py` 中维护；改映射后需重跑评估。
- **文档**：`BENCHMARK_PROGRESS.md` §0 摘要、`benchmark_plan.md` §5.3/§8 已同步；论文主表指向上述两个 comparison/by_token CSV。

## 2026-05-24 训练分工与 CTS 口径确认（用户确认）

- **背景**：澄清各分支训练落点，避免 `CLAUDE.md` 仍写「V5B 待跑」「V5A2/B2 在服务器」。
- **项目内容与结论**：
  - **V5B**：在**服务器**完成全量训练；本机保留 `v5b/best`（tar）供 benchmark / 推理。
  - **V5A2 / V5B2**：原拟**服务器** LoRA 独立目录；后确认**本机 V5A1/V5B1** 即可跑完整 LoRA，**不再另训 A2/B2**（非磁盘缺失）。
  - **CTS**：环境转化已通过 **patRoon BT 流程**完成并写入 `patoon_cts_*.csv`，**不再单独跑 EPA CTS API**。
- **关键约束**：有效训练分支为 **四线**：V5A、V5B（服务器）+ V5A1、V5B1（本机 LoRA）；文档与 benchmark 勿再等待 A2/B2。
- **后续提醒**：`train_v5a2.py` / `train_v5b2.py` 仅作将来需隔离目录时的可选入口。

## 2026-05-25 Gradio 单条预测 Top10 候选展示

- **项目内容与结论**：
  - `scratch/app_gradio.py` 单条预测从只展示 PPM 重排 top-1，扩展为同时输出 **Top10 候选表**。
  - 候选表按 PPM / 质量误差升序排序，含 SMILES、产物精确质量、Da/mDa 误差、PPM、PPM 是否通过、合法性、可选真值 Exact/Tanimoto。
  - 高精度模式会把 TTA 生成候选、TTA 共识与 direct beam 候选合并后统一去重排序；快速模式直接生成 10 个 beam 候选。
- **最佳实践**：
  - 对质量约束场景，不能只看 top-1；Top10 能判断“正确候选是否曾被生成但排序靠后”，还是“候选池根本没有正确结构”。
  - 候选去重使用有效结构的 canonical SMILES，invalid SMILES 保留在表尾用于诊断生成质量。
- **已确认的坑与规避方式**：
  - 现有 Gradio 进程不会自动加载代码修改，需重启 `scratch/app_gradio.py` 后界面才出现 Top10 表。
  - 仓库 venv 未安装 `pytest`，新增轻量回归测试使用标准库 `unittest`：`python scratch/test_app_gradio_topk.py`。

## 2026-05-25 Gradio MCS 公共结构/变化区域高亮

- **项目内容与结论**：
  - `scratch/app_gradio.py` 单条预测新增 **公共结构 / 变化区域（MCS 高亮）** 图片输出。
  - 使用 RDKit `rdFMCS.FindMCS` 识别母体与主预测 top-1 的最大公共子结构；绿色显示公共结构，红色显示变化原子/键。
  - 保留原有普通“母体 → 预测产物结构”图，新增 MCS 图仅用于解释主预测，不对 Top10 全部候选逐个绘图。
- **最佳实践**：
  - MCS 图适合诊断模型是否保留核心骨架，以及反应位点是否落在预期区域；对 atrazine 等 OOD 案例可快速看到是否丢 Cl、改环或侧链变化错误。
  - 若预测 SMILES 无效或 MCS 不可靠，函数返回空图，避免展示误导性高亮。
- **验证**：
  - `python scratch/test_app_gradio_topk.py` 覆盖 Top10 排序/去重与 MCS 图像生成。
  - `python -m py_compile scratch/app_gradio.py scratch/test_app_gradio_topk.py` 通过；linter 无报错。

## 2026-05-25 MCS 高亮颜色修复

- **问题**：用户截图发现 MCS 图几乎全是红色，公共结构未按绿色显示。
- **根因**：RDKit `Draw.MolToImage` 在当前环境下没有可靠应用 `highlightAtomColors` / `highlightBondColors`，导致自定义多色高亮退化为默认红色；MCS 匹配本身正常（atrazine -> hydroxyatrazine 可匹配 13 个公共原子）。
- **修复**：`scratch/app_gradio.py` 的 `_draw_highlighted_mol` 改用 `rdMolDraw2D.MolDraw2DCairo.DrawMolecule`，显式传入逐原子/逐键颜色。
- **验证**：`scratch/test_app_gradio_topk.py` 新增颜色回归测试，要求 atrazine -> 2-hydroxyatrazine 案例中绿色像素数大于红色像素数；`python scratch/test_app_gradio_topk.py` 4 项通过。

## 2026-06-16 V6a 训练部署与 API 兼容性修复

- **背景**：本机启动 V6a 训练三次均因 transformers 5.7.0 API 变更提前终止，同时全量训练 ~150h 不适合本机执行，改为服务器部署。
- **现象/报错**：
  1. `ImportError: cannot import name 'WEIGHTS_NAME' from 'transformers.training_args'`
  2. `TypeError: Seq2SeqTrainer.__init__() got an unexpected keyword argument 'tokenizer'`
- **根因**：transformers 5.7.0 中 `WEIGHTS_NAME` 移至 `transformers.utils`，`TRAINING_ARGS_NAME` 被移除；`Seq2SeqTrainer` 将 `tokenizer` 参数重命名为 `processing_class`。
- **处理动作**：
  - `WEIGHTS_NAME` 改为 `from transformers.utils import WEIGHTS_NAME`，`TRAINING_ARGS_NAME` 硬编码为 `"training_args.bin"`；
  - `V6Trainer(..., tokenizer=...)` → `processing_class=...`；
  - 新增 `SC_CLM_SERVER=1` 环境变量控制服务器/本机双模式（batch/workers/checkpointing）；
  - 新增 FileHandler 将日志持久化到 `logs/v6a/train_*.log`；
  - 训练已移交服务器，本机清理所有失败日志。
- **最佳实践**：
  - 升级 transformers 大版本后先做 import smoke test，不直接启动长训练；
  - 服务器/本机双模式用环境变量切换，避免维护两套 train.py。
- **关键约束**：
  - V6a 训练仅服务器执行，本机保留代码/测试/数据做快速验证；
  - 服务器 checkpoint 需手动同步回本机。
- **已确认的坑与规避方式**：
  - `warmup_ratio` 和 `logging_dir` 在 transformers 5.x 已 deprecated，后续需改用 `warmup_steps` 和 `TENSORBOARD_LOGGING_DIR`；
  - `processing_class` 不仅是改名——旧版 `tokenizer` kwarg 会直接报 TypeError，不是 warning。

## 2026-06-15 V6a/V6b 双线模型实现完成

- **背景**：按 `scratch/v6_parallel_instructions.md` 并行实现 V6a（ReactionT5+mass embedding）与 V6b（OpenNMT+mass token）。
- **项目内容与结论**：
  - V6a 沿 V5SCLM 架构模式，将 18 个离散反应 token 替换为 532-bin 量化质量差嵌入（`nn.Embedding(532, 768)`），CFG 仍在编码器级别插值，LoRA 同 V5B1；
  - V6b 为全新框架：OpenNMT Transformer 6L、字符级分词（Kekulé SMILES）、质量 token 以输入前缀形式注入（`MASS_{bin}`）；
  - 数据合并 raw (8,156) + CTS LIKELY (3,948) → 去重后 8,361 train / 412 val / 152 test；
  - 测试：V6a 26/26 pass / V6b 15/15 pass。
- **最佳实践**：
  - 质量量化方案作为两个模型的共享基础设施，V6b config 自包含复制（不 import torch 链）避免框架版本冲突；
  - evaluate.py 框架在模型缺失时优雅跳过，支持增量评估；
  - `mass_to_bin_tensor` 委托给标量 `mass_to_bin` 循环实现，保证一致性并避免 `torch.bucketize` 浮点边界问题。
- **关键约束**：
  - ReactionT5 `d_model=768` 不是 512，mass_embed_dim 必须匹配；
  - opennmt-py 3.5.1 硬约束 `torch<2.3`，与 transformers 要求的 `torch>=2.4` 冲突 → V6b 训练需独立 conda 环境；
  - 安装了 opennmt-py 会导致 torch 自动降级，破坏 V6a/transformers 可用性 → 已回退，当前 venv 保持 torch 2.5.1。
- **已确认的坑与规避方式**：
  - `pip install opennmt-py` 静默降级 torch 2.5.1→2.2.2，且伴随 numpy 2.x 不兼容 → **不要在主 venv 安装 opennmt-py**，用独立环境；
  - 向量化 `torch.bucketize` 与二分查找 `mass_to_bin` 在 bin 边界值上可能不一致 → 统一走标量函数循环；
  - 带 BOM 的 R 脚本会报 `unexpected invalid token in "﻿"` → 用 `sed -i '1s/^\xEF\xBB\xBF//'` 剥离。
- **后续提醒**：训练启动前先验证 `models/ReactionT5v2-forward` 本地快照可用，避免 HF 不可达阻塞。

## 2026-05-26 Gradio 质量模式 PPM 合格候选全量展示

- **项目内容与结论**：
  - `scratch/app_gradio.py` 在 **Monoisotopic Mass 模式** 下新增 **`PPM≤10 合格候选（全部列出）`** 表格。
  - 主预测优先取自 **PPM 合格候选** 中误差最小者；若无合格候选，回退为 Top10 中 PPM 最优。
  - 质量模式下 beam 扩至 20，以增大候选池；仍保留 Top10 总表（含未通过 PPM 者）供诊断。
- **验证**：`python scratch/test_app_gradio_topk.py` 6 项通过。

## 2026-06-16 V7 预研方向讨论（等 V6 训练完成后启动）

- **背景**：V6a/V6b 已部署服务器训练，讨论下一版架构改进方向。任务本质是条件分子生成（parent SMILES + delta_mz → product SMILES）。
- **V5 现状**：V5B PPM 72.48%, Exact 30.28%, Tanimoto 0.653。长 SMILES 子集 PPM 降至 54~61%。外部 superbio PPM 83% 但 Exact 弱。
- **V7 候选方向（按 ROI 排序）**：
  - **P0 V7a: Cross-Attention Mass Conditioning** — mass→MLP→decoder cross-attention 替代 encoder prepend，参考 Stable Diffusion 条件注入。+ 连续 MLP(1→256→768) 替代 532-bin 离散 Embedding。预期 PPM +3~8%，改动量小。
  - **P1 SELFIES** — 100% 有效分子串消除 validity 问题，需重建 tokenizer/vocab。
  - **P1 对比预训练** — (parent, product) 对 + 质量差判别信号，弥补 8K 数据不足。
  - **P1 多任务学习** — 同时预测产物 + 反应类型，改善高频 token。
  - **P2 GNN Encoder + Transformer Decoder** — 图表示保留拓扑结构，高回报但复杂度高。
  - **P2 Flow Matching / Diffusion** — 前沿但不稳定，不推荐当前阶段。
  - **应用层** — 多模型集成 V5B/V6a + CTS/BT 投票/精排。
- **最佳实践**：V7a 改动集中在 model.py（cross-attention 注入）和 config.py（MLP 替 Embedding），训练/推理脚本基本复用 V6a。
- **关键约束**：不改 ReactionT5 骨干（已验证有效），不改 LoRA 策略，8GB VRAM 红线。
- **执行条件**：服务器 V6a/b 训练完成 + evaluate.py 出对比表后，根据结果决定 V7 优先级。
