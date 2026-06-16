import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def split_test_for_routing(test_csv_path: str, seed: int = 42) -> tuple:
    """
    将测试集严格分为 50% 路由校准集 (Calibration) 和 50% 最终评估集 (Evaluation)，
    并且在划分时维持同类 Scaffold 不跨越两个集（零泄露原则的延伸）。
    但既然已经是测试集，母体SMILES不同即可，为保证Token分布一致，最好按Token进行分层抽样。
    """
    df = pd.read_csv(test_csv_path)
    logger.info(f"读取原始测试集: {len(df)} 行")
    
    # 按照 Token 分层，随机划分为 50/50
    calib_list = []
    eval_list = []
    
    rng = np.random.default_rng(seed)
    
    for token, group in df.groupby('token'):
        n = len(group)
        indices = group.index.values.copy()
        rng.shuffle(indices)
        
        mid = n // 2
        calib_idx = indices[:mid]
        eval_idx = indices[mid:]
        
        calib_list.append(group.loc[calib_idx])
        eval_list.append(group.loc[eval_idx])
        
    df_calib = pd.concat(calib_list).sample(frac=1, random_state=seed).reset_index(drop=True)
    df_eval = pd.concat(eval_list).sample(frac=1, random_state=seed).reset_index(drop=True)
    
    logger.info(f"路由校准集 (Calibration): {len(df_calib)} 行")
    logger.info(f"最终评估集 (Evaluation): {len(df_eval)} 行")
    
    return df_calib, df_eval

def main():
    test_path = 'data/processed/test.csv'
    if not Path(test_path).exists():
        logger.error(f"{test_path} 不存在，请先完成数据预处理。")
        return
        
    df_calib, df_eval = split_test_for_routing(test_path)
    
    Path('data/routing').mkdir(parents=True, exist_ok=True)
    df_calib.to_csv('data/routing/test_calibration.csv', index=False)
    df_eval.to_csv('data/routing/test_evaluation.csv', index=False)
    
    logger.info("Adaptive Routing 测试集拆分完成，数据已保存至 data/routing/")
    logger.info("下一步：")
    logger.info("1. 用训练好的模型在 test_calibration.csv 上分别测试 No-Token 和 有-Token 的推理表现。")
    logger.info("2. 构建基于统计显著性 (n>=15, diff>10pp) 的动态路由表。")
    logger.info("3. 在 test_evaluation.csv 上使用该路由表进行最终零泄露评估。")

if __name__ == '__main__':
    main()
