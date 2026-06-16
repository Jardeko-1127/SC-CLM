# CC Implementation Instructions: V6a Model

你需要在 E:\DA\DeepMet\SC_CLM 项目中实现 V6a 模型。以下是精确的架构设计和实现要求。

## 架构概览

V6a = ReactionT5v2-forward + 量化质量差Embedding + CFG + LoRA (V5B1风格)

核心改动：将V5的10个离散反应token替换为~510个量化质量差bin。

## 质量差量化方案

基于 12,000 pairs 的 delta_mz 分布：
- 核心区 [-100, 200]: 1 Da bins → 300 bins (覆盖76.1%)
- 外围 [-500, -100) ∪ [200, 500]: 5 Da bins → 140 bins (覆盖21.6%)
- 极值 (< -500 或 ≥ 500): 10 Da bins → ~70 bins
- 总计 ~510 bins

## 需要创建的文件

### 1. src/model/v6/__init__.py
空文件或导出主要类。

### 2. src/model/v6/config.py
```python
@dataclass
class V6Config:
    base_model: str = "sagawa/ReactionT5v2-forward"  # 或本地路径
    mass_embed_dim: int = 512  # 必须匹配 ReactionT5 d_model
    
    # Mass quantization config
    mass_bins: list = [...]  # 预计算的bin边界和中心值
    
    # LoRA config (V5B1 style)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list = ["q", "k", "v", "o"]
    
    # CFG
    cfg_dropout_prob: float = 0.15
    cfg_guidance_scale: float = 1.5
    
    # Training (V5B1 style for 8GB VRAM)
    micro_batch: int = 1
    grad_accum: int = 8
    learning_rate: float = 2e-4
    num_epochs: int = 30
    max_input_len: int = 256
    max_output_len: int = 128
    seed: int = 42
    
    # Paths
    train_csv: str = "data/processed/v6_train.csv"
    val_csv: str = "data/processed/v6_val.csv"
    output_dir: str = "results/checkpoints/v6a"
    log_dir: str = "logs/v6a"
    
    # PPM
    ppm_threshold: float = 5.0
    
    def mass_to_bin(self, delta_mz: float) -> int:
        """量化delta_mz到最近的bin索引"""
        ...
    
    def bin_to_mass(self, bin_idx: int) -> float:
        """bin索引→delta_mz中心值"""
        ...
```

Mass bins的具体定义（在config中硬编码或从JSON加载）：

核心区[-100, 200]@1Da: bins=[-100, -99, ..., 199] → 300 bins，bin中心 = bin值 + 0.5
外围[-500, -100)@5Da: bins=[-500, -495, ..., -105] → 80 bins，bin中心 = bin值 + 2.5
外围[200, 500]@5Da: bins=[200, 205, ..., 495] → 60 bins，bin中心 = bin值 + 2.5
极值<-500@10Da: bins=[-1070, -1060, ..., -510] → 57 bins，bin中心 = bin值 + 5
极值>=500@10Da: bins=[500, 510, ..., 840] → 35 bins，bin中心 = bin值 + 5

总计: 300 + 140 + 92 = 532 bins

### 3. src/model/v6/model.py
V6SCLM 类，核心架构：

```python
class V6SCLM(nn.Module):
    """
    V6 Mass-Conditioned Chemical Language Model (V6a).
    
    Architecture:
      parent SMILES → ReactionT5 Encoder → hidden (B, L, 512)
      delta_mz (float) → quantize → bin_idx → Embedding[bin_idx] → (B, 1, 512)
      hidden + prepend(mass_embed) → Conditioned Hidden (B, L+1, 512)
      ReactionT5 Decoder → product SMILES
    """
    
    def __init__(self, config: V6Config):
        # 1. 加载 ReactionT5v2-forward backbone
        # 2. 创建 Mass Embedding: nn.Embedding(num_mass_bins, 512)
        # 3. 初始化 embedding 权重 (normal distribution)
        # 4. 应用 LoRA (peft)
    
    def _get_mass_embed(self, delta_mz: torch.Tensor, training: bool):
        # 量化 delta_mz → bin_idx
        # 查 embedding
        # CFG dropout: 以概率p置零
    
    def _encode(self, input_ids, attention_mask, delta_mz, training):
        # ReactionT5 encoder
        # prepend mass_embed
        # 返回 conditioned hidden + extended mask
    
    def forward(self, input_ids, attention_mask, labels, delta_mz):
        # 标准 seq2seq forward
    
    def generate_with_cfg(self, input_ids, attention_mask, delta_mz, guidance_scale):
        # CFG: guided = uncond + γ × (cond - uncond)
    
    def save(self, path, tokenizer=None):
        # 保存 backbone (含LoRA) + mass_embed + config
    
    @classmethod
    def load(cls, path, tokenizer=None):
        # 加载完整模型
```

关键实现细节：
- delta_mz 是一个 (B,) 的 float tensor
- 使用 torch.bucketize 或自定义函数将连续值映射到bin索引
- mass_embed 是 nn.Embedding(num_bins, 512)，初始化用 nn.init.normal_(std=0.02)
- CFG dropout: 以cfg_dropout_prob将mass_embed置零（作为无条件模式）
- LoRA: 使用 peft.LoraConfig + get_peft_model，target_modules=["q","k","v","o"]，task_type=TaskType.SEQ_2_SEQ_LM
- 注意：LoRA 不能应用在 mass_embed 上，只应用在 base_model 的 attention 层

### 4. src/model/v6/dataset.py
V6Dataset 类：

```python
class V6Dataset(Dataset):
    """
    返回:
      - input_ids: tokenized parent SMILES
      - attention_mask
      - labels: tokenized product SMILES  
      - delta_mz: 浮点数质量差
      - bin_idx: 量化后的bin索引 (长整型)
    """
    
    def __init__(self, csv_path, tokenizer, mass_config, ...):
        # csv 包含列: parent_smiles, product_smiles, delta_mz
        # 计算 bin_idx = mass_config.mass_to_bin(delta_mz)
        # 支持 SMILES augmentation (同V5)
        # 支持 chemical_whitespace (同V5)
```

### 5. src/model/v6/prepare_data.py
数据准备脚本，将 raw + CTS 合并为统一训练集：

```python
def prepare_v6_data():
    """
    1. 从 data/raw/ 加载所有CSV (排除 S81_THSTPS)
    2. 从 results/cts_augmented/cts_all_likely_*.csv 加载CTS结果
    3. 合并，计算 delta_mz
    4. RDKit验证
    5. InChIKey-14去重
    6. Scaffold分层划分 train/val/test
    7. 输出到 data/processed/v6_train.csv, v6_val.csv, v6_test.csv
    """
```

关键：
- 排除 S81_THSTPS_Transformations.csv（三手烟，无关）
- 标记来源列 source: 'raw' 或 'cts_photolysis' / 'cts_hydrolysis' / 'cts_abiotic_reduction'
- CSV 列: parent_smiles, product_smiles, delta_mz, source

### 6. src/model/v6/train.py
训练脚本（V5B1 LoRA 风格）：

```python
def main():
    # 1. 加载 config
    # 2. 创建 tokenizer (使用 ReactionT5 tokenizer 或 T5Tokenizer)
    #    - 添加 chemical whitespace 相关的 special tokens
    # 3. 创建 V6SCLM 模型
    # 4. 应用 LoRA (peft)
    # 5. 加载 train/val dataset
    # 6. 创建 V6DataCollator (同V5风格)
    # 7. 创建 V6Trainer (subclass Seq2SeqTrainer)
    #    - 需要传递 delta_mz 到 model.generate()
    # 8. TrainingArguments:
    #    - per_device_train_batch_size=1
    #    - gradient_accumulation_steps=8
    #    - fp16=True
    #    - gradient_checkpointing=True
    #    - eval_strategy="epoch", save_strategy="epoch"
    #    - predict_with_generate=True
    #    - generation_max_length=128
    # 9. 训练 + 保存 best model
```

注意：
- 训练时将 delta_mz 作为模型输入传递
- V6Trainer 需要 override prediction_step 来传递 delta_mz
- 验证时计算 PPM fidelity（使用 target_mass 作为分母）

### 7. src/model/v6/inference.py
推理脚本：

```python
def predict(parent_smiles: str, product_mz: float, model, tokenizer, config):
    """
    输入: 母体SMILES + 产物 m/z
    输出: 候选产物SMILES列表
    
    流程:
    1. 计算 parent_mass from SMILES
    2. delta_mz = product_mz - parent_mass
    3. 量化到bin → mass_embed
    4. CFG generate (guidance_scale=1.5)
    5. 解码 → 产物SMILES
    6. 验证产物有效性
    """
```

### 8. tests/test_v6a.py
测试文件，测试以下内容：

```python
class TestV6MassQuantization:
    def test_quantization_core(self):
        """核心区 1Da 量化"""
        # 15.3 → bin for 15
        # -14.7 → bin for -15
    
    def test_quantization_outer(self):
        """外围 5Da 量化"""
        # 230 → bin for 230
        # -333 → bin for -335
    
    def test_quantization_extreme(self):
        """极值 10Da 量化"""
        # -888 → bin for -890

class TestV6Model:
    def test_model_creation(self):
        """创建 V6SCLM 不报错"""
    
    def test_forward_pass(self):
        """单次前向传播不报错，输出形状正确"""
    
    def test_cfg_generation(self):
        """CFG生成产出有效SMILES"""
    
    def test_save_load(self):
        """保存和加载往返一致"""

class TestV6Dataset:
    def test_dataset_loading(self):
        """Dataset正确读取CSV并返回delta_mz"""

class TestV6Inference:
    def test_end_to_end(self):
        """parent SMILES + m/z → product SMILES"""
```

## 实现顺序

1. 先写 `prepare_data.py` → 运行生成 v6_train/val/test.csv
2. 写 `config.py` → 质量量化函数
3. 写 `model.py` → V6SCLM 核心
4. 写 `dataset.py` → V6Dataset
5. 写 `train.py` → V6Trainer
6. 写 `inference.py` → 推理接口
7. 写 `tests/test_v6a.py` → 跑测试

## 关键约束

- ReactionT5v2-forward 本地路径: models/ReactionT5v2-forward（如果有），否则用 HuggingFace "sagawa/ReactionT5v2-forward"
- 本地模型加载时注意 torch.load CVE bypass（参考 V5 代码）
- LoRA 只加在 attention 层 (q_proj, k_proj, v_proj, o_proj)
- mass_embed 不参与 LoRA
- 所有代码使用与 V5 一致的编码风格（中文注释、logger等）
- 使用项目 venv: .\venv\Scripts\python.exe

完成后运行 python tests/test_v6a.py 验证所有测试通过。
