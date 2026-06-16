# CC Parallel Implementation: V6a + V6b

你需要同时推进两个模型的实现。两者共享质量量化方案和数据整合逻辑，
但模型框架完全独立。

## V6a (先做)

按 scratch/v6a_cc_instructions.md 执行：
1. 创建 src/model/v6/config.py (532 bins量化)
2. 创建 src/model/v6/prepare_data.py → 运行生成 v6_train/val/test.csv
3. 创建 src/model/v6/model.py → V6SCLM
4. 创建 src/model/v6/dataset.py → V6Dataset
5. 创建 src/model/v6/train.py → V6Trainer (LoRA)
6. 创建 src/model/v6/inference.py
7. 创建 tests/test_v6a.py → 运行验证

## V6b (后用V6a的config，独立实现)

V6b 必须从 src.model.v6.config 导入 V6Config 复用质量量化。

按 scratch/v6b_cc_instructions.md 执行：
1. pip install opennmt-py ctranslate2 pyonmttok (安装到venv)
2. 创建 src/model/v6b/config.py (from v6.config import V6Config)
3. 创建 src/model/v6b/prepare_data.py → 复用v6a数据整合，输出OpenNMT格式
4. 创建 src/model/v6b/build_vocab.py
5. 创建 src/model/v6b/train.py (生成YAML + onmt_train)
6. 创建 src/model/v6b/inference.py (ctranslate2)
7. 创建 src/model/v6b/evaluate.py (V6a vs V6b对比)
8. 创建 tests/test_v6b.py → 运行验证

## 执行顺序

Step 1: V6a config.py (先做，V6b依赖它)
Step 2: V6a prepare_data.py → 运行生成数据
Step 3: V6a model.py + dataset.py + train.py + inference.py
Step 4: V6a tests → 验证通过
Step 5: V6b 全部 (可并行于Step 3-4，但必须在Step 1-2完成后开始)

## 关键提醒

- V6b 是独立框架 (OpenNMT)，不是 V6a 的变体
- V6b 训练用 onmt_train CLI，不是 HuggingFace Trainer
- 两个模型最终在 evaluate.py 中用同一测试集对比
