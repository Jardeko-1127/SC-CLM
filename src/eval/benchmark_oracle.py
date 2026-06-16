import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm
from pathlib import Path
import sys
import os

# Project Root Setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.eval.metrics import V4Metrics

# ── Config ──
TRAIN_CSV = "data/processed/train.csv"
TEST_CSV = "data/processed/test.csv"
OUTPUT_CSV = "results/benchmark_oracle_v4.csv"
PPM_THRESHOLD = 5.0
MASS_TOL_DA = 0.01

class OracleBenchmark:
    """
    m/z Retrieval Baseline (Oracle).
    Searches the training product library for the closest mass match.
    Represents the database lookup performance limit.
    """
    def __init__(self):
        self.metrics = V4Metrics()
        self.db = []
        self.masses = np.array([])

    def build_db(self, train_path):
        print(f"Building Oracle DB from {train_path}...")
        df = pd.read_csv(train_path)
        unique_targets = df['product_smiles'].unique().tolist()
        
        valid_db = []
        for smi in tqdm(unique_targets, desc="Calculating masses"):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    valid_db.append({
                        "smiles": smi,
                        "mass": Descriptors.ExactMolWt(mol),
                        "inchikey": self.metrics.get_inchikey_14(smi)
                    })
            except: pass
            
        valid_db.sort(key=lambda x: x['mass'])
        self.db = valid_db
        self.masses = np.array([x['mass'] for x in valid_db])
        print(f"Oracle DB Size: {len(self.db)} unique molecules.")

    def run_benchmark(self, test_path):
        if not self.db:
            self.build_db(TRAIN_CSV)
            
        df_test = pd.read_csv(test_path)
        results = []
        
        print(f"Running Oracle Benchmark on {len(df_test)} samples...")
        for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
            parent = row['parent_smiles']
            target = row['product_smiles']
            delta = row['delta_mz']
            
            p_mol = Chem.MolFromSmiles(parent)
            if not p_mol: continue
            
            expected_mass = Descriptors.ExactMolWt(p_mol) + delta
            
            # Find closest mass in DB
            idx = np.searchsorted(self.masses, expected_mass)
            
            # Check neighborhood
            best_smi = ""
            best_diff = float('inf')
            
            # Search a window of 200 around the insertion point
            for i in range(max(0, idx - 100), min(len(self.masses), idx + 100)):
                diff = abs(self.masses[i] - expected_mass)
                if diff < best_diff and diff <= MASS_TOL_DA:
                    best_diff = diff
                    best_smi = self.db[i]['smiles']
            
            target_ik = self.metrics.get_inchikey_14(target)
            best_ik = self.metrics.get_inchikey_14(best_smi) if best_smi else None
            
            results.append({
                "token": row['token'],
                "parent_smiles": parent,
                "target_smiles": target,
                "prediction": best_smi,
                "em": (best_ik == target_ik) if best_ik and target_ik else False,
                "tanimoto": self.metrics.calc_tanimoto(target, best_smi) if best_smi else 0.0,
                "ppm_pass": self.metrics.check_ppm_fidelity(parent, best_smi, delta) if best_smi else False
            })
            
        df_res = pd.DataFrame(results)
        df_res.to_csv(OUTPUT_CSV, index=False)
        
        print("\nOracle Benchmark Summary:")
        print(f"Top-1 EM: {df_res['em'].mean()*100:.2f}%")
        print(f"PPM Pass: {df_res['ppm_pass'].mean()*100:.2f}%")
        print(f"Tanimoto Median: {df_res['tanimoto'].median():.4f}")

if __name__ == "__main__":
    oracle = OracleBenchmark()
    oracle.run_benchmark(TEST_CSV)
