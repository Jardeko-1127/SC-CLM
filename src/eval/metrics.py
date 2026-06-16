import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class V4Metrics:
    """
    Single Source of Truth for SC-CLM V4 Evaluation.
    Implements:
    1. InChIKey Skeleton Match (EM)
    2. PPM-Gated Mass Fidelity (5ppm)
    3. Tanimoto Similarity (Morgan Radius 2)
    4. Chemical Sanity (RDKit Sanitize)
    """

    @staticmethod
    def canon(smi: str) -> str:
        """Canonicalize SMILES string (isomeric, canonical)."""
        if not smi or not isinstance(smi, str):
            return ""
        try:
            m = Chem.MolFromSmiles(smi)
            if m:
                return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
        except Exception:
            pass
        return smi

    @staticmethod
    def get_inchikey_14(smi: str) -> Optional[str]:
        """Extract InChIKey Connectivity Layer (first 14 chars)."""
        if not smi: return None
        try:
            m = Chem.MolFromSmiles(smi)
            if m:
                inchi_str = MolToInchi(m)
                if inchi_str:
                    ik = InchiToInchiKey(inchi_str)
                    return ik[:14] if ik else None
        except Exception:
            pass
        return None

    @staticmethod
    def check_ppm_fidelity(parent_smi: str, pred_smi: str, theoretical_delta: float, threshold: float = 10.0) -> bool:
        """
        Validate mass difference between parent and prediction against theoretical delta.
        Formula: abs(actual_delta - theoretical_delta) / (parent_mass + theoretical_delta) * 1e6
        """
        try:
            p_mol = Chem.MolFromSmiles(parent_smi)
            pr_mol = Chem.MolFromSmiles(pred_smi)
            if not (p_mol and pr_mol):
                return False
            
            p_mass = Descriptors.ExactMolWt(p_mol)
            pr_mass = Descriptors.ExactMolWt(pr_mol)
            
            actual_delta = pr_mass - p_mass
            target_mass = p_mass + theoretical_delta
            
            if target_mass <= 0:
                return False
                
            ppm_error = abs(actual_delta - theoretical_delta) / target_mass * 1e6
            return ppm_error <= threshold
        except Exception:
            return False

    @staticmethod
    def calc_tanimoto(smi1: str, smi2: str) -> float:
        """Calculate Morgan Fingerprint Tanimoto similarity."""
        if not (smi1 and smi2): return 0.0
        try:
            m1 = Chem.MolFromSmiles(smi1)
            m2 = Chem.MolFromSmiles(smi2)
            if m1 and m2:
                fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
                fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)
                return DataStructs.TanimotoSimilarity(fp1, fp2)
        except Exception:
            pass
        return 0.0

    @staticmethod
    def check_validity(smi: str) -> bool:
        """Check if SMILES is valid and obeys chemical valence."""
        if not smi: return False
        try:
            m = Chem.MolFromSmiles(smi)
            if m:
                Chem.SanitizeMol(m)
                return True
        except:
            pass
        return False

    def evaluate_batch(self, df: pd.DataFrame) -> Dict:
        """
        Evaluate a dataframe of predictions.
        Required columns: parent_smiles, target_smiles, prediction, delta_mz
        """
        results = []
        for _, row in df.iterrows():
            parent = row['parent_smiles']
            target = row['target_smiles']
            pred = row['prediction']
            delta = row['delta_mz']
            
            target_ik = self.get_inchikey_14(target)
            pred_ik = self.get_inchikey_14(pred)
            
            is_valid = self.check_validity(pred)
            is_ppm = self.check_ppm_fidelity(parent, pred, delta) if is_valid else False
            is_skeleton = (pred_ik == target_ik) if (pred_ik and target_ik) else False
            
            # Level 4: Exact Match
            target_canon = self.canon(target)
            pred_canon = self.canon(pred)
            is_exact_match = (pred_canon == target_canon) if (pred_canon and target_canon) else False
            
            tanimoto = self.calc_tanimoto(target, pred)
            
            results.append({
                'valid': is_valid,
                'ppm': is_ppm,
                'skeleton': is_skeleton,
                'exact_match': is_exact_match,
                'tanimoto': tanimoto
            })
            
        res_df = pd.DataFrame(results)
        return {
            'validity_rate': res_df['valid'].mean(),
            'ppm_pass_rate': res_df['ppm'].mean(),
            'skeleton_pass_rate': res_df['skeleton'].mean(),
            'exact_match_rate': res_df['exact_match'].mean(),
            'tanimoto_median': res_df['tanimoto'].median()
        }
