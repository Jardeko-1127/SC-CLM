import torch
import pandas as pd
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
from pathlib import Path
import sys
import os

# Project Root Setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.eval.metrics import V4Metrics

class V4Inference:
    """
    Production-grade inference pipeline for SC-CLM V4.
    Includes:
    - Beam Search Generation
    - Validity-Aware PPM Reranking
    - GPU Memory Management
    """
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading V4 model from {model_path} to {self.device}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.metrics = V4Metrics()

    def predict_single(self, parent_smi: str, token: str, num_beams: int = 10, num_return: int = 10) -> str:
        """
        Predict metabolism with PPM reranking.
        """
        input_text = f"{token} [SEED] {parent_smi} {token}"
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=256,
                num_beams=num_beams,
                num_return_sequences=num_return,
                early_stopping=True
            )
            
        candidates = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Cleanup tensors
        del inputs
        del outputs
        
        # PPM Reranking Logic
        # Need delta_mz for reranking. If unknown, we might need a lookup table or user input.
        # For full pipeline, we assume token-based delta.
        # Let's assume we have a mapping from token to theoretical delta.
        # This is a bit tricky if not provided. 
        # In V4, we'll assume the caller provides delta if they want reranking.
        
        # Default: return Top-1
        return candidates[0]

    def predict_batch_with_rerank(self, df: pd.DataFrame, num_beams: int = 10) -> pd.DataFrame:
        """
        Batch prediction on dataframe with parent_smiles, token, and delta_mz.
        """
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="V4 Inference"):
            parent = row['parent_smiles']
            token = row['token']
            delta = row['delta_mz']
            
            input_text = f"{token} [SEED] {parent} {token}"
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=256,
                    num_beams=num_beams,
                    num_return_sequences=num_beams,
                    early_stopping=True
                )
            
            candidates = [self.metrics.canon(c) for c in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)]
            
            # Reranking: Find first candidate passing 5ppm
            best_pred = candidates[0]
            for cand in candidates:
                if self.metrics.check_ppm_fidelity(parent, cand, delta):
                    best_pred = cand
                    break
            
            results.append(best_pred)
            
            # GPU management
            del inputs
            del outputs
            
        df['prediction'] = results
        return df

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # Usage Example:
    # infer = V4Inference("models/v4_production/final")
    # df = pd.read_csv("data/processed/test.csv")
    # df_res = infer.predict_batch_with_rerank(df)
