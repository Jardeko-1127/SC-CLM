# CC Implementation Instructions: V6b (MetaReact-inspired)

## 背景

分析 MetaReact (https://github.com/myzhengSIMM/MetaReact.git) 后，提取其有价值的架构思路，
创建 V6b：基于 OpenNMT Transformer + 质量差条件注入的环境转化产物预测模型。

V6b 与 V6a (ReactionT5-based) 形成架构对照实验。

## V6b 架构设计

### 核心创新：质量差条件注入 + Transformer

```
输入: parent_SMILES | delta_mz_bin
  ↓
字符级tokenization (空格分隔，同MetaReact)
  ↓
OpenNMT Transformer Encoder → Decoder
  ↓
产物SMILES (Kekulé or canonical)
```

### 与 MetaReact 的区别
1. MetaReact输入: "substrate | <unk>" → 我们的输入: "parent_SMILES | delta_mz"
2. MetaReact输出: 代谢产物 → 我们输出: 环境转化产物
3. 用我们的 raw+CTS 数据训练，不是MetaReact的预训练权重
4. 添加质量差token作为条件

## 需要创建的文件

### 1. src/model/v6b/__init__.py
空文件

### 2. src/model/v6b/config.py
V6bConfig:
```python
@dataclass
class V6bConfig:
    # Model
    num_layers: int = 6
    num_heads: int = 8
    hidden_size: int = 512
    ffn_size: int = 2048
    dropout: float = 0.1
    
    # Mass bins (与V6a共享量化方案)
    mass_bins_count: int = 532
    mass_token_offset: int = 1000  # 质量token的起始ID
    
    # Training
    batch_size: int = 4096  # tokens
    learning_rate: float = 2.0
    num_epochs: int = 30
    max_src_len: int = 500
    max_tgt_len: int = 500
    
    # Paths
    data_dir: str = "data/processed/v6b"
    model_dir: str = "results/checkpoints/v6b"
    log_dir: str = "logs/v6b"
```

### 3. src/model/v6b/prepare_data.py
数据准备（输出为OpenNMT格式）:

```python
def prepare_v6b_data():
    """
    1. 复用 v6a 的数据整合逻辑（raw+CTS合并）
    2. 计算 delta_mz
    3. 量化到 bin index
    4. 输出OpenNMT格式:
       src-train.txt: parent_SMILES | MASS_{bin_idx}
       tgt-train.txt: product_SMILES (kekulized)
       src-val.txt, tgt-val.txt
    5. 构建OpenNMT YAML配置
    """
```

关键：SMILES转为Kekulé格式（用RDKit Kekulize），因为MetaReact使用Kekulé格式训练。
格式: `C 1 = C C = C C = C 1 O | M A S S _ 1 5`

### 4. src/model/v6b/build_vocab.py
构建词汇表:

```python
def build_vocab():
    """
    从训练数据构建pyonmttok词汇表
    字符级: C, 1, =, (, ), [, ], @, /, \, ., +, -, #, %, 0-9, ...
    质量token: M, A, S, _, 0-9
    特殊token: |, <unk>, <s>, </s>, <blank>
    """
```

### 5. src/model/v6b/train.py
训练脚本:

```python
def train_v6b():
    """
    使用OpenNMT-py命令行训练:
    
    onmt_train \
      -config v6b_config.yaml \
      -save_model results/checkpoints/v6b/model
    """
```

需要生成正确的YAML配置:
```yaml
save_data: data/processed/v6b/opennmt
src_vocab: data/processed/v6b/src_vocab.txt
tgt_vocab: data/processed/v6b/tgt_vocab.txt

encoder_type: transformer
decoder_type: transformer
layers: 6
heads: 8
hidden_size: 512
word_vec_size: 512
transformer_ff: 2048
dropout: 0.1

batch_size: 4096
batch_type: tokens
normalization: tokens
accum_count: 4
optim: adam
learning_rate: 2.0
warmup_steps: 8000
decay_method: noam

train_steps: 50000
valid_steps: 2000
save_checkpoint_steps: 2000
keep_checkpoint: 5
seed: 42
```

### 6. src/model/v6b/inference.py
推理脚本:

```python
def predict_v6b(parent_smiles: str, product_mz: float, model_path: str):
    """
    1. 计算 delta_mz, 量化到 bin
    2. 构建输入: parent_SMILES | MASS_{bin}
    3. OpenNMT translate (beam=20, n_best=20)
    4. 后处理: kekulize→canonical, 去重, 按score排序
    5. 返回 top-k 候选产物 (k=5 or 10)
    """
```

### 7. src/model/v6b/evaluate.py
评估脚本，对比V6a和V6b:

```python
def compare_v6a_v6b():
    """
    在同一测试集上评估:
    - V6a (ReactionT5 + mass embedding)
    - V6b (Transformer + mass token)
    
    指标: validity, exact match, PPM fidelity, Tanimoto
    """
```

### 8. tests/test_v6b.py
测试:
```python
class TestV6bDataPrep:
    def test_data_format(self):
    def test_mass_token_encoding(self):

class TestV6bInference:
    def test_predict_single(self):
    def test_compare_with_v6a(self):
```

## 实现顺序

1. `config.py` - 配置定义
2. `prepare_data.py` - 数据准备 → 生成OpenNMT格式
3. `build_vocab.py` - 词汇表构建  
4. `train.py` - 训练配置 + 启动脚本
5. `inference.py` - 推理
6. `evaluate.py` - V6a/V6b对比
7. `tests/test_v6b.py` - 测试

## 关键决策

- **OpenNMT安装**: `pip install opennmt-py==3.2.0 pyonmttok ctranslate2` 到venv
- **不用MetaReact预训练权重**: 用我们的 ~12,000 pairs 从头训练
- **Kekulé格式**: 训练时使用，推理后转回canonical SMILES
- **质量token**: 编码为 "MASS_{bin_idx}" 如 "MASS_15", "MASS_NEG_500"
- **beam=20**: 同MetaReact，生成多个候选
- **batch_type=tokens**: 自动批处理，适合8GB显存

## 对比实验设计

| 维度 | V6a | V6b |
|------|-----|-----|
| Backbone | ReactionT5v2 (~80M) | Transformer (6L, ~45M) |
| 分词 | BPE subword | Character-level |
| 条件注入 | Embedding prepend | Input token prefix |
| 推理策略 | CFG (γ=1.5) | Beam search (b=20) |
| 候选数 | 1-5 (beam) | 20 (n_best) |
| 参数量 | ~80M | ~45M |

完成后运行 tests/test_v6b.py 验证数据准备和推理流程。
