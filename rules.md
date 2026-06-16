# SC-CLM 项目规则（精简统一版）

本文件用于统一项目工程规则。若与分散规则有冲突，以本文件为准。

## 1) 代码规范

- 遵循 PEP8
- 公共函数与方法必须使用类型注解
- 采用 Google 风格 docstring
- 路径处理使用 `pathlib`
- 变量/函数使用 `snake_case`，类名使用 `PascalCase`

## 2) 化学领域规范

- 所有输入/输出 SMILES 必须经过 `Chem.MolFromSmiles` 校验
- 处理 `MolFromSmiles(...) is None` 的失败分支，不得静默忽略
- 分子质量统一使用 `Descriptors.ExactMolWt`
- 禁止字符串拼接方式做质量加减
- 默认以 InChIKey-14 做去重与精确匹配标识
- 评估中 PPM 分母必须使用 `target_mass`

## 3) 训练与工程规范

- 超参数放配置或 argparse，避免硬编码
- 生产脚本使用 `logging`，避免 `print`
- 训练前设置并记录随机种子（`random/numpy/torch/transformers`）
- 检查 CUDA 可用性，不可用时自动回退 CPU
- checkpoint 输出到版本化目录（如 `results/checkpoints/v5a`）

## 4) 协作与文档规范

- 主要协作窗口使用 `CLAUDE.md`，并保持中文
- 每次重大变更后同步更新 `CLAUDE.md` 与必要的 README
- 历史资料归档到 `ref/`，避免根目录堆积过期文档
- 统一维护 `project_memory.md`，并遵循“踩坑即时记录 + 任务完成总结”流程

## 5) Skills 使用策略

- 优先使用项目相关 skills（`skills/`、`.claude/agents/`）
- `.agents/skills/` 下与项目无关技能按需读取，不主动加载

## 6) 改动洁癖审查

- 每次代码改动前后都做一轮 over-engineering 自检
- 若实现存在过度设计，立即简化为更直接、更稳妥的版本
- 在不牺牲正确性的前提下，优先可读性与可维护性

## 7) 会话交接规范（handoff）

- 在“离开/关闭 CC 窗口”或“context 接近 60%”时，必须更新 `handoff.md`
- `handoff.md` 必须使用固定结构：
  - `## 当前在做什么`
  - `## 已经试过的方案和结果(含失败的)`
  - `## 下一步计划(3-5条 actionable)`
  - `## 关键文件路径(相对路径，一行一个)`
  - `## 还没搞清楚的问题`
- 目标：3 天后 30 秒内恢复工作上下文

## 8) 日常回复结构（可选）

- 平时对话不强制固定模板，按任务复杂度简洁表达即可
- 若任务复杂，建议使用 5 段结构提升可追踪性
- 唯一强制要求是按第 7 条在触发点更新 `handoff.md`

