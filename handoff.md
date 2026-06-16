## 2026-06-16 V6 三模型并行训练 & V7路线确定

### 服务器硬件（3× RTX 3090 24GB）
- GPU 0: V6a 训练中（67% util, 5.4GB）
- GPU 1: V6b 卡死在数据加载（Weighted corpora循环，0% util）
- GPU 2: 空闲，待上V6c

### V6a/b/c 关系
| 模型 | 骨干 | 质量注入 | 解码 | 状态 |
|------|------|---------|------|------|
| V6a | ReactionT5v2 | 532-bin prepend | CFG+LoRA | 训练中 |
| V6b | OpenNMT 6L | MASS token prefix | Beam search | 卡死待修 |
| V6c | ReactionT5v2 | 532-bin/MLP append | CFG+LoRA | 待创建 |

V6c=V6a的唯一变体：mass注入位置prepend→append，其余全部复用。

### V6b问题：batch_type=tokens导致字符级SMILES数据加载死循环
修复：batch_type=sentences, batch_size=64, accum_count=2

### V7路线（CC评估7方向后确定）
Phase 1: 多模型集成 + V7a(V6c Cross-Attn+MLP)
Phase 2: 对比预训练 + 多任务学习
Phase 3: GNN+Transformer
不投入: SELFIES, Flow Matching

### 关键文件
- handoff.md — 项目总览+V7预研方向
- project_memory.md — 新增V7讨论条目
- scratch/v7_candidate_analysis.html — CC评估报告
