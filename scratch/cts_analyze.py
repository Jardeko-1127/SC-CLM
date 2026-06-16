# ============================================================
# CTS结果 + raw数据 整合统计分析脚本
# 用法：CTS跑完后，在PowerShell中运行：
#   .\venv\Scripts\python.exe scratch/cts_analyze.py
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors
from collections import Counter, defaultdict
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

RAW_DIR = Path("data/raw")
CTS_DIR = Path("results/cts_augmented")

# ============================================================
# 1. 加载所有raw数据
# ============================================================
print("=" * 60)
print("1. 加载 raw 数据")
print("=" * 60)

raw_pairs = []  # (parent_smi, product_smi, transformation, source)
raw_parents = set()

for f in sorted(RAW_DIR.glob("*.csv")):
    try:
        df = pd.read_csv(f, encoding='utf-8-sig')
    except:
        try:
            df = pd.read_csv(f, encoding='latin-1')
        except:
            continue
    
    if 'Predecessor_SMILES' not in df.columns:
        continue
    if 'Successor_SMILES' not in df.columns:
        continue
    
    trans_col = 'Transformation' if 'Transformation' in df.columns else None
    
    for _, row in df.iterrows():
        p_smi = str(row['Predecessor_SMILES']).strip()
        s_smi = str(row['Successor_SMILES']).strip()
        trans = str(row[trans_col]).strip() if trans_col else 'unknown'
        
        p_mol = Chem.MolFromSmiles(p_smi)
        s_mol = Chem.MolFromSmiles(s_smi)
        if not (p_mol and s_mol):
            continue
        
        p_can = Chem.MolToSmiles(p_mol, canonical=True, isomericSmiles=True)
        s_can = Chem.MolToSmiles(s_mol, canonical=True, isomericSmiles=True)
        
        raw_pairs.append({
            'parent_smiles': p_can,
            'product_smiles': s_can,
            'transformation': trans,
            'source': f.name
        })
        raw_parents.add(p_can)

raw_df = pd.DataFrame(raw_pairs)
print(f"  Raw valid pairs: {len(raw_df)}")
print(f"  Raw unique parent SMILES: {len(raw_parents)}")
print(f"  Raw unique product SMILES: {raw_df['product_smiles'].nunique()}")
print(f"  Raw unique (parent,product) pairs: {raw_df.drop_duplicates(['parent_smiles','product_smiles']).shape[0]}")

# ============================================================
# 2. 加载 CTS LIKELY 结果
# ============================================================
print("\n" + "=" * 60)
print("2. 加载 CTS 结果")
print("=" * 60)

# 找最新的 all_likely 文件
likely_files = sorted(CTS_DIR.glob("cts_all_likely_*.csv"))
if not likely_files:
    # 尝试找各个pathway的likely文件
    likely_files = sorted(CTS_DIR.glob("cts_likely_*.csv"))

if likely_files:
    cts_file = likely_files[-1]  # 最新的
    cts_df = pd.read_csv(cts_file)
    print(f"  CTS file: {cts_file.name}")
    print(f"  CTS LIKELY candidates: {len(cts_df)}")
    
    cols = cts_df.columns.tolist()
    print(f"  Columns: {cols}")
    
    # 统计
    if 'parent_SMILES' in cts_df.columns:
        cts_parents = set(cts_df['parent_SMILES'].dropna())
        print(f"  CTS parents with LIKELY products: {len(cts_parents)}")
    
    if 'cts_pathway' in cts_df.columns:
        print(f"\n  By pathway:")
        for pw in cts_df['cts_pathway'].dropna().unique():
            sub = cts_df[cts_df['cts_pathway'] == pw]
            n_parents = sub['parent_SMILES'].nunique() if 'parent_SMILES' in sub.columns else 0
            print(f"    {pw}: {len(sub)} products, {n_parents} parents")
else:
    print("  WARNING: No CTS LIKELY files found!")
    cts_df = None

# ============================================================
# 3. 对比 raw vs CTS 覆盖
# ============================================================
print("\n" + "=" * 60)
print("3. Raw vs CTS 覆盖对比")
print("=" * 60)

all_unique_parents = set()
for f in sorted(RAW_DIR.glob("*.csv")):
    try:
        df = pd.read_csv(f, encoding='utf-8-sig')
    except:
        try:
            df = pd.read_csv(f, encoding='latin-1')
        except:
            continue
    if 'Predecessor_SMILES' not in df.columns:
        continue
    for smi in df['Predecessor_SMILES'].dropna():
        mol = Chem.MolFromSmiles(str(smi).strip())
        if mol:
            all_unique_parents.add(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))

print(f"  全部 unique 母体 (raw): {len(all_unique_parents)}")

if cts_df is not None and 'parent_SMILES' in cts_df.columns:
    cts_parent_set = set(cts_df['parent_SMILES'].dropna())
    overlap = all_unique_parents & cts_parent_set
    raw_only = all_unique_parents - cts_parent_set
    cts_only = cts_parent_set - all_unique_parents
    
    print(f"  CTS 覆盖的母体: {len(cts_parent_set)}")
    print(f"  Raw+CTS 重叠母体: {len(overlap)}")
    print(f"  仅在Raw中的母体: {len(raw_only)}")
    print(f"  仅在CTS中有LIKELY产物的母体: {len(cts_only)}")

# ============================================================
# 4. 反应类型分布
# ============================================================
print("\n" + "=" * 60)
print("4. Raw 数据反应类型分布 (Top 20)")
print("=" * 60)

trans_counts = raw_df['transformation'].value_counts()
for i, (t, c) in enumerate(trans_counts.head(20).items()):
    pct = c / len(raw_df) * 100
    print(f"  {str(t)[:60]:60s} {c:5d} ({pct:4.1f}%)")
print(f"  ... total unique transformation types: {raw_df['transformation'].nunique()}")

# ============================================================
# 5. 质量差分布
# ============================================================
print("\n" + "=" * 60)
print("5. 质量差分布 (parent -> product Delta MW)")
print("=" * 60)

def calc_delta_mw(row):
    p_mol = Chem.MolFromSmiles(row['parent_smiles'])
    s_mol = Chem.MolFromSmiles(row['product_smiles'])
    if p_mol and s_mol:
        return round(Descriptors.ExactMolWt(s_mol) - Descriptors.ExactMolWt(p_mol), 4)
    return None

raw_df['delta_mw'] = raw_df.apply(calc_delta_mw, axis=1)
deltas = raw_df['delta_mw'].dropna()

print(f"  Mean: {deltas.mean():.2f}")
print(f"  Std: {deltas.std():.2f}")
print(f"  Min: {deltas.min():.2f}")
print(f"  Max: {deltas.max():.2f}")
print(f"  Median: {deltas.median():.2f}")

# 分桶统计
bins = [float('-inf'), -100, -50, -20, -5, 0, 5, 20, 50, 100, 200, 500, float('inf')]
labels = ['<-100', '-100~-50', '-50~-20', '-20~-5', '-5~0', '0~5', '5~20', '20~50', '50~100', '100~200', '200~500', '>500']
raw_df['delta_bin'] = pd.cut(raw_df['delta_mw'], bins=bins, labels=labels)
print(f"\n  Delta MW bins:")
for bin_label, cnt in raw_df['delta_bin'].value_counts().sort_index().items():
    print(f"    {bin_label:>12s}: {cnt:5d}")

# ============================================================
# 6. 母体来源统计
# ============================================================
print("\n" + "=" * 60)
print("6. 数据来源分布")
print("=" * 60)

source_counts = raw_df['source'].value_counts()
for src, cnt in source_counts.items():
    parents_in_src = raw_df[raw_df['source'] == src]['parent_smiles'].nunique()
    print(f"  {src}: {cnt} pairs, {parents_in_src} unique parents")

# ============================================================
# 7. CTS产物中的转化类型（如果能提取）
# ============================================================
if cts_df is not None:
    print("\n" + "=" * 60)
    print("7. CTS 产物统计")
    print("=" * 60)
    
    if 'SMILES' in cts_df.columns:
        valid_cts = cts_df[cts_df['SMILES'].apply(lambda x: Chem.MolFromSmiles(str(x)) is not None)]
        print(f"  Valid product SMILES: {len(valid_cts)} / {len(cts_df)}")
    
    if 'parent_SMILES' in cts_df.columns and 'SMILES' in cts_df.columns:
        # 计算 CTS 的质量差
        cts_deltas = []
        for _, row in cts_df.iterrows():
            p_mol = Chem.MolFromSmiles(str(row['parent_SMILES']))
            s_mol = Chem.MolFromSmiles(str(row['SMILES']))
            if p_mol and s_mol:
                cts_deltas.append(Descriptors.ExactMolWt(s_mol) - Descriptors.ExactMolWt(p_mol))
        
        if cts_deltas:
            cts_d = pd.Series(cts_deltas)
            print(f"\n  CTS delta MW stats:")
            print(f"    Mean: {cts_d.mean():.2f}")
            print(f"    Std: {cts_d.std():.2f}")
            print(f"    Min: {cts_d.min():.2f}")
            print(f"    Max: {cts_d.max():.2f}")

print("\nDONE")
