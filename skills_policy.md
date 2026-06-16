# Skills 使用策略（SC-CLM）

## 目标

避免与本项目无关的通用学术写作 skills 干扰模型研发与训练协作上下文。

## 当前策略

- 项目主线默认只使用：
  - `skills/`（本项目自定义化学技能）
  - `.claude/agents/`（本项目子代理）
- 原 `.agents/skills/` 已归档到：
  - `ref/archive_skills/skills/`

## 按需取用

当且仅当出现明确需求（如论文写作、审稿、Typst/LaTeX 支持）时，再临时从 `ref/archive_skills/skills/` 读取对应 `SKILL.md`。

## 恢复方式

如需恢复原位置，可将：

- `ref/archive_skills/skills/`

移动回：

- `.agents/skills/`
