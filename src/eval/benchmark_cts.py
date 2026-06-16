import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import time
import sys
import json
import os
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Project Root Setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.eval.metrics import V4Metrics

# ── Config ──
BASE_URL = "https://qed.epa.gov/cts/rest"
CACHE_FILE = "data/cache/cts_api_cache.json"
MAX_WORKERS = 5 # Reduced to avoid overwhelming API and race conditions

TOKEN_TO_CTS_LIBS = {
    "[TRANS_HYDRATION]":       ["hydrolysis"],
    "[TRANS_OXIDATION]":       ["photolysis", "mammalian_metabolism"],
    "[TRANS_N-OXIDATION]":     ["mammalian_metabolism"],
    "[TRANS_DEMETHYLATION]":   ["mammalian_metabolism"],
    "[TRANS_DEETHYLATION]":    ["mammalian_metabolism"],
    "[TRANS_ACETYLATION]":     ["mammalian_metabolism"],
    "[TRANS_GLUCURONIDATION]": ["mammalian_metabolism"],
    "[TRANS_DEHYDROGENATION]": ["mammalian_metabolism"],
    "[TRANS_ISOMERIZATION]":   ["hydrolysis"],
    "[TRANS_DI_OXIDATION]":    ["mammalian_metabolism"],
    "[TRANS_DEHALOGENATION]":  ["abiotic_reduction", "anaerobic_biodegradation"],
    "[TRANS_DESISOPROPYL]":    ["mammalian_metabolism"],
    "[TRANS_DIDESMETHYL]":     ["mammalian_metabolism"],
}

class CTSBenchmark:
    def __init__(self):
        self.metrics = V4Metrics()
        self.cache_lock = threading.Lock()
        self.cache = self._load_cache()
        self.session = self._get_session()

    def _load_cache(self):
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r") as f:
                    return json.load(f)
            except: pass
        return {}

    def _save_cache(self):
        with self.cache_lock:
            Path(CACHE_FILE).parent.mkdir(parents=True, exist_ok=True)
            with open(CACHE_FILE, "w") as f:
                json.dump(self.cache, f, indent=2)

    def _get_session(self):
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=3, status_forcelist=[502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def extract_smiles(self, node, parent_smiles, candidates):
        if not node: return
        smiles = node.get("data", {}).get("smiles", "")
        if smiles and smiles != parent_smiles:
            candidates.add(smiles)
        for child in node.get("children", []):
            self.extract_smiles(child, parent_smiles, candidates)

    def query_cts(self, smiles, libs):
        all_found = set()
        for lib in libs:
            cache_key = f"{smiles}::{lib}"
            with self.cache_lock:
                if cache_key in self.cache:
                    all_found.update(self.cache[cache_key])
                    continue

            payload = {
                "structure": smiles,
                "generationLimit": 1,
                "populationLimit": 0,
                "lifeStages": ["adult"],
                "transformationLibraries": [lib],
            }
            try:
                resp = self.session.post(f"{BASE_URL}/metabolizer/run", json=payload, timeout=120)
                if resp.status_code == 200:
                    data = resp.json()
                    root = data.get("data", {}).get("data", {})
                    lib_smiles = set()
                    self.extract_smiles(root, smiles, lib_smiles)
                    
                    with self.cache_lock:
                        self.cache[cache_key] = list(lib_smiles)
                        self._save_cache()
                    all_found.update(lib_smiles)
                time.sleep(2) # Global-ish rate limit (per thread, but combined with MAX_WORKERS)
            except Exception as e:
                print(f"CTS Error for {smiles} ({lib}): {e}")
        return list(all_found)

    def process_row(self, row):
        parent, target, token, delta = row['parent_smiles'], row['product_smiles'], row['token'], row['delta_mz']
        libs = TOKEN_TO_CTS_LIBS.get(token, ["mammalian_metabolism"])
        
        candidates = self.query_cts(parent, libs)
        
        # PPM Reranking using V4 standard
        best_pred = ""
        if candidates:
            scored = []
            for c in candidates:
                if self.metrics.check_ppm_fidelity(parent, c, delta):
                    scored.append(c)
            # If PPM passes, pick first (CTS doesn't provide scores/probabilities in a simple way here)
            best_pred = scored[0] if scored else candidates[0]

        target_ik = self.metrics.get_inchikey_14(target)
        best_ik = self.metrics.get_inchikey_14(best_pred)
        
        return {
            "token": token,
            "parent_smiles": parent,
            "target_smiles": target,
            "prediction": best_pred,
            "em": (best_ik == target_ik) if best_ik and target_ik else False,
            "coverage": len(candidates) > 0,
            "tanimoto": self.metrics.calc_tanimoto(target, best_pred),
            "n_candidates": len(candidates)
        }

    def run_benchmark(self, input_csv, output_csv):
        df = pd.read_csv(input_csv)
        results = []
        
        print(f"Starting CTS V4 Benchmark on {len(df)} samples...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(self.process_row, row): i for i, row in df.iterrows()}
            for future in tqdm(as_completed(futures), total=len(df)):
                results.append(future.result())
        
        df_res = pd.DataFrame(results)
        df_res.to_csv(output_csv, index=False)
        
        print("\nCTS Benchmark Summary:")
        print(f"Coverage: {df_res['coverage'].mean()*100:.2f}%")
        print(f"Top-1 EM: {df_res['em'].mean()*100:.2f}%")
        print(f"Tanimoto Median: {df_res['tanimoto'].median():.4f}")

if __name__ == "__main__":
    bench = CTSBenchmark()
    # Path logic should align with data/processed/test.csv
    bench.run_benchmark("data/processed/test.csv", "results/benchmark_cts_v4.csv")
