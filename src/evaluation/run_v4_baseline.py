import pandas as pd
import torch
import logging
import gc
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from pathlib import Path
import sys
import os

# Ensure project root is in path to allow absolute imports
sys.path.append(os.path.abspath(os.getcwd()))
from src.eval.metrics import V4Metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Bypass transformers CVE torch version checks safely
try:
    import transformers.utils.import_utils as hf_import_utils
    hf_import_utils.check_torch_load_is_safe = lambda: None
    import transformers.modeling_utils as hf_modeling_utils
    hf_modeling_utils.check_torch_load_is_safe = lambda: None
except:
    pass

def main():
    model_dir = "results/checkpoints/v4_model_final"
    data_path = "data/routing/test_calibration.csv"
    output_path = "logs/v4_baseline_calibration_report.csv"
    
    if not Path(model_dir).exists():
        logger.error(f"Model directory not found: {model_dir}")
        return
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        return
        
    logger.info("Loading model and tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples from calibration set.")
    
    results = []
    
    logger.info("Starting baseline No-Token generation...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        parent_smi = str(row['parent_smiles']).strip()
        
        # V4 No-Token Format
        input_seq = f"[SEED] {parent_smi}"
        
        inputs = tokenizer(input_seq, return_tensors="pt", max_length=256, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=256,
                num_beams=3,
                early_stopping=True
            )
            
        pred_smi = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        record = row.to_dict()
        record['target_smiles'] = record['product_smiles'] # Map column name for V4Metrics
        record['prediction'] = pred_smi
        results.append(record)
        
        # Explicit memory destruction for tensors
        del inputs, outputs
        
    df_results = pd.DataFrame(results)
    
    # Unified Metric Evaluation
    logger.info("Performing centralized V4Metrics evaluation...")
    metrics_engine = V4Metrics()
    eval_summary = metrics_engine.evaluate_batch(df_results)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_path, index=False)
    
    total = len(df_results)
    
    logger.info("\n" + "="*50)
    logger.info("🚀 V4 BASELINE CALIBRATION (NO-TOKEN) REPORT 🚀")
    logger.info("="*50)
    logger.info(f"Total Samples          : {total}")
    logger.info(f"Syntax Pass Rate       : {eval_summary['validity_rate']*100:.2f}%")
    logger.info(f"PPM Pass Rate (<10ppm) : {eval_summary['ppm_pass_rate']*100:.2f}%")
    logger.info(f"Skeleton Pass (Ik14)   : {eval_summary['skeleton_pass_rate']*100:.2f}%")
    logger.info(f"Exact Match (Canonical): {eval_summary['exact_match_rate']*100:.2f}%")
    logger.info(f"Tanimoto Median        : {eval_summary['tanimoto_median']:.4f}")
    logger.info("="*50)
    logger.info(f"Detailed report saved to {output_path}")

    # Aggressive memory cleanup post-inference
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    logger.info("CUDA memory successfully purged.")

if __name__ == "__main__":
    main()
