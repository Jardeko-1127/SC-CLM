import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Draw
from pathlib import Path
import logging
import sys
import os

# Project Root Setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.eval.metrics import V4Metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class V4ErrorAnalyzer:
    """
    Error Pathology Analysis for SC-CLM V4.
    Categorizes errors into positional isomers, functional group replacements, etc.
    Generates visual gallery of errors.
    """
    def __init__(self, report_dir="logs/v4_error_analysis"):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.img_dir = self.report_dir / "gallery"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = V4Metrics()

    def get_formula(self, smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            return rdMolDescriptors.CalcMolFormula(mol) if mol else None
        except: return None

    def run_pathology(self, results_csv: str, model_label: str = "V4_Model"):
        logger.info(f"Running error pathology for {model_label}...")
        df = pd.read_csv(results_csv)
        
        # Required columns: parent_smiles, target_smiles, prediction, delta_mz, em
        if 'em' not in df.columns:
            logger.error("Dataframe must contain 'em' (Exact Match) column.")
            return

        errors = df[df['em'] == False].copy()
        total_errors = len(errors)
        if total_errors == 0:
            logger.info("No errors found. Perfect performance?")
            return

        pathology_results = []
        for i, row in errors.iterrows():
            target = row['target_smiles']
            pred = row['prediction']
            
            t_ik = self.metrics.get_inchikey_14(target)
            p_ik = self.metrics.get_inchikey_14(pred)
            t_formula = self.get_formula(target)
            p_formula = self.get_formula(pred)
            tanimoto = self.metrics.calc_tanimoto(target, pred)
            
            p_type = "Other"
            if p_formula == t_formula and t_ik != p_ik and tanimoto > 0.7:
                p_type = "Type A: Positional Isomer"
            elif p_formula != t_formula and tanimoto > 0.7:
                p_type = "Type B: FG Replacement"
            elif tanimoto < 0.3:
                p_type = "Type D: Total Hallucination"
            elif self.metrics.check_ppm_fidelity(row['parent_smiles'], pred, row['delta_mz']):
                p_type = "Type C: Logic Fidelity Orphan"

            pathology_results.append(p_type)
            
            # Generate image for top 20 errors
            if len(pathology_results) <= 20:
                self.save_error_image(row, p_type, len(pathology_results))

        errors['pathology'] = pathology_results
        
        # Summary report
        report_path = self.report_dir / f"pathology_report_{model_label}.md"
        summary = errors['pathology'].value_counts()
        
        with open(report_path, "w") as f:
            f.write(f"# V4 Error Pathology Report: {model_label}\n\n")
            f.write(f"- Total Samples: {len(df)}\n")
            f.write(f"- Total Errors: {total_errors}\n")
            f.write(f"- Error Rate: {total_errors/len(df)*100:.2f}%\n\n")
            f.write("## Distribution\n\n")
            f.write("| Pathology Type | Count | Percentage |\n")
            f.write("| --- | --- | --- |\n")
            for p_type, count in summary.items():
                f.write(f"| {p_type} | {count} | {count/total_errors*100:.2f}% |\n")
        
        logger.info(f"Pathology report saved to {report_path}")

    def save_error_image(self, row, p_type, idx):
        try:
            m_target = Chem.MolFromSmiles(row['target_smiles'])
            m_pred = Chem.MolFromSmiles(row['prediction'])
            if m_target and m_pred:
                img = Draw.MolsToGridImage(
                    [m_target, m_pred],
                    molsPerRow=2,
                    subImgSize=(400, 300),
                    legends=["Ground Truth", f"Pred ({p_type})"]
                )
                img_path = self.img_dir / f"error_{idx}.png"
                img.save(img_path)
        except: pass

if __name__ == "__main__":
    analyzer = V4ErrorAnalyzer()
    # analyzer.run_pathology("results/v4_test_results.csv")
