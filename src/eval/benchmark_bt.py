import pandas as pd
import numpy as np
import subprocess
import shutil
import os
import sys
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Project Root Setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.eval.metrics import V4Metrics

# ── Config ──
BT_JAR = os.path.join(PROJECT_ROOT, "tools", "biotransformer", "target", "biotransformer-3.0.0.jar")
BT_DIR = os.path.join(PROJECT_ROOT, "tools", "biotransformer")

def _detect_java():
    if "JAVA_HOME" in os.environ:
        candidate = os.path.join(os.environ["JAVA_HOME"], "bin", "java.exe")
        if os.path.isfile(candidate): return candidate
    found = shutil.which("java")
    return found if found else "java"

JAVA_EXE = _detect_java()

TOKEN_MODE_MAP = {
    "[TRANS_N-OXIDATION]":    ["allHuman", "env"],
    "[TRANS_DEMETHYLATION]":  ["allHuman", "env"],
    "[TRANS_DIDESMETHYL]":    ["allHuman", "env"],
    "[TRANS_ACETYLATION]":    ["allHuman"],
    "[TRANS_DEHALOGENATION]": ["env"],
    "[TRANS_DESISOPROPYL]":   ["env"],
}
DEFAULT_MODES = ["env"]

class BioTransformerBenchmark:
    def __init__(self):
        self.metrics = V4Metrics()
        self.cache_lock = threading.Lock()
        self.cache = {}

    def run_bt_single(self, smiles, mode, thread_id=0):
        cache_key = (smiles, mode)
        with self.cache_lock:
            if cache_key in self.cache: return self.cache[cache_key]

        out_csv = os.path.join(BT_DIR, f"bt_tmp_{mode}_{thread_id}.csv")
        if os.path.exists(out_csv):
            try: os.remove(out_csv)
            except: pass

        cmd = [JAVA_EXE, "-jar", BT_JAR, "-k", "pred", "-b", mode, "-ismi", smiles, "-ocsv", out_csv, "-s", "1"]
        
        try:
            subprocess.run(cmd, cwd=BT_DIR, capture_output=True, text=True, timeout=180)
        except: return []

        products = []
        if os.path.exists(out_csv):
            try:
                df = pd.read_csv(out_csv)
                if "SMILES" in df.columns:
                    for s in df["SMILES"].dropna():
                        c = self.metrics.canon(str(s))
                        if c: products.append(c)
            except: pass
            
        with self.cache_lock:
            self.cache[cache_key] = list(set(products))
        return self.cache[cache_key]

    def process_row(self, row, thread_id):
        parent, target, token, delta = row['parent_smiles'], row['product_smiles'], row['token'], row['delta_mz']
        modes = TOKEN_MODE_MAP.get(token, DEFAULT_MODES)
        
        all_candidates = []
        for mode in modes:
            all_candidates.extend(self.run_bt_single(parent, mode, thread_id))
            
        all_candidates = list(set(all_candidates))
        
        # PPM Reranking
        best_pred = ""
        if all_candidates:
            scored = []
            for c in all_candidates:
                if self.metrics.check_ppm_fidelity(parent, c, delta):
                    scored.append(c)
            best_pred = scored[0] if scored else all_candidates[0]

        target_ik = self.metrics.get_inchikey_14(target)
        best_ik = self.metrics.get_inchikey_14(best_pred)
        
        return {
            "token": token,
            "parent_smiles": parent,
            "target_smiles": target,
            "prediction": best_pred,
            "em": (best_ik == target_ik) if best_ik and target_ik else False,
            "coverage": len(all_candidates) > 0,
            "tanimoto": self.metrics.calc_tanimoto(target, best_pred)
        }

    def run_benchmark(self, input_csv, output_csv):
        if not os.path.exists(BT_JAR):
            print(f"BioTransformer JAR not found at {BT_JAR}. Skipping BT benchmark.")
            return

        df = pd.read_csv(input_csv)
        results = []
        
        print(f"Starting BioTransformer 3.0 V4 Benchmark on {len(df)} samples...")
        with ThreadPoolExecutor(max_workers=4) as executor: # Keep low to avoid JVM memory contention
            futures = {executor.submit(self.process_row, row, i): i for i, row in df.iterrows()}
            for future in tqdm(as_completed(futures), total=len(df)):
                results.append(future.result())
                
        df_res = pd.DataFrame(results)
        df_res.to_csv(output_csv, index=False)
        
        print("\nBioTransformer Benchmark Summary:")
        print(f"Coverage: {df_res['coverage'].mean()*100:.2f}%")
        print(f"Top-1 EM: {df_res['em'].mean()*100:.2f}%")
        print(f"Tanimoto Median: {df_res['tanimoto'].median():.4f}")

if __name__ == "__main__":
    bench = BioTransformerBenchmark()
    bench.run_benchmark("data/processed/test.csv", "results/benchmark_bt_v4.csv")
