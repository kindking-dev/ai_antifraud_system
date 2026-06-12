#!/usr/bin/env python3
"""
Sentinel AI: Master Thesis Appendix Workspace Compiler.
Executes setup, generates all evaluation visualizations, and exports 
API schemas, SHAP outputs, logs, and LaTeX guide files.
"""

import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"

SCRIPTS = [
    "setup_appendix.py",
    "generate_appendix_data.py",
    "generate_dataset_distribution.py",
    "generate_pr_curve.py",
    "generate_latency_distribution.py",
    "generate_threshold_matrix.py",
    "generate_shap_visualizations.py",
]

def run_script(script_name: str):
    print(f"\n" + "="*70)
    print(f"[*] Running: {script_name}")
    print(f"="*70)
    
    script_path = REPORTS_DIR / script_name
    try:
        # Run with the same Python interpreter
        res = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"[+] Success: {script_name} completed.")
    except subprocess.CalledProcessError as e:
        print(f"[!] Error: {script_name} failed with exit code {e.returncode}")
        sys.exit(e.returncode)

def main():
    print("="*70)
    print("[*] SENTINEL AI: THESIS APPENDIX BUILD PIPELINE STARTED")
    print("="*70)
    
    # Run setup first, then generate data, then plots
    for script in SCRIPTS:
        run_script(script)
        
    print("\n" + "="*70)
    print("[+] ALL THESIS APPENDIX MATERIALS GENERATED AND COMPILED SUCCESSFULLY!")
    print("="*70)
    print(f"Figures folder : {REPORTS_DIR / 'thesis_figures'}")
    print(f"API/Logs folder: {REPORTS_DIR / 'appendix_exports'}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
