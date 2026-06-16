import pandas as pd
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def optimize_routing_table(calib_results_path: str, output_json: str):
    """
    Analyzes calibration results to build an optimal routing table.
    Criteria:
    - n >= 15 for statistical significance.
    - If Full_EM > NoTok_EM + 0.10 (10pp improvement) -> route to 'full'
    - Otherwise -> route to 'no_token'
    """
    df = pd.read_csv(calib_results_path)
    
    # We expect columns like: token, full_em, notok_em (boolean flags)
    # If the input is from run_v4_baseline (NoTok) and a future run_v4_full, we merge them.
    # For now, let's assume a combined CSV.
    
    tokens = df['token'].unique()
    routing_table = {}
    
    print(f"\n{'Token':<28} {'N':>4} {'Full EM':>10} {'NoTok EM':>10} {'Decision':>10}")
    print("-" * 65)
    
    for token in sorted(tokens):
        sub = df[df['token'] == token]
        n = len(sub)
        
        # In a real scenario, we'd have both columns. 
        # If not, we default to no_token unless proven otherwise.
        full_em = sub['full_em'].mean() if 'full_em' in sub.columns else 0.0
        notok_em = sub['notok_em'].mean() if 'notok_em' in sub.columns else 0.0
        
        decision = 'no_token'
        if n >= 15 and (full_em - notok_em) > 0.10:
            decision = 'full'
            
        routing_table[token] = decision
        print(f"{token:<28} {n:>4} {full_em*100:>9.1f}% {notok_em*100:>9.1f}% {decision:>10}")

    with open(output_json, "w") as f:
        json.dump(routing_table, f, indent=2)
    logger.info(f"Routing table saved to {output_json}")

if __name__ == "__main__":
    # This requires the calibration run to be completed
    # optimize_routing_table("logs/v4_calibration_merged.csv", "configs/v4_routing_table.json")
    pass
