#!/usr/bin/env python3
"""
AI Anti-Fraud Service: Academic Precision-Recall (PR) Curve Generator.
Loads validation predictions, computes precision/recall curves,
and exports an IEEE-style evaluation plot (PNG & PDF) ready for LaTeX.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
VAL_PREDS_PATH = BASE_DIR / "reports" / "val_predictions.csv"
OUTPUT_DIR = BASE_DIR / "reports" / "thesis_figures"
OUTPUT_PNG = OUTPUT_DIR / "precision_recall_curve.png"
OUTPUT_PDF = OUTPUT_DIR / "precision_recall_curve.pdf"

# Matplotlib styling for Academic/IEEE publication
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 12,
    "text.usetex": False,  # Portability fallback
})

def main():
    # 1. Load predictions
    if not VAL_PREDS_PATH.exists():
        print(f"[!] Error: Prediction file not found at '{VAL_PREDS_PATH}'.")
        print("    Please run the model evaluation or ensure 'val_predictions.csv' is generated.")
        sys.exit(1)
        
    print(f"[*] Loading validation predictions from {VAL_PREDS_PATH}...")
    df = pd.read_csv(VAL_PREDS_PATH)
    
    y_true = df["y_true"]
    y_pred = df["y_pred"]
    
    # 2. Compute Metrics
    print("[*] Computing Precision-Recall metrics...")
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    ap_score = average_precision_score(y_true, y_pred)
    no_skill = len(y_true[y_true == 1]) / len(y_true)
    
    print(f"    - Class Balance (Fraud ratio): {no_skill:.4f}")
    print(f"    - Average Precision (AP) Score: {ap_score:.4f}")
    
    # 3. Plotting
    print("[*] Rendering Precision-Recall curve...")
    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=300)
    ax.set_facecolor("white")
    
    # Plot PR Curve (Dark gray/black solid line for monochrome print safety)
    ax.plot(
        recall, 
        precision, 
        color="#111111", 
        linewidth=1.8, 
        label=f"CatBoost Fraud Detector (AP = {ap_score:.4f})"
    )
    
    # Plot Random Baseline (Dashed gray line)
    ax.plot(
        [0, 1], 
        [no_skill, no_skill], 
        color="#777777", 
        linestyle="--", 
        linewidth=1.2, 
        label=f"Random Baseline (AP = {no_skill:.4f})"
    )
    
    # Clean axes spines (IEEE style: remove top and right borders)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    
    # Labels & Title
    ax.set_title("AI Anti-Fraud Service: Precision-Recall (PR) Curve", pad=15)
    ax.set_xlabel("Recall (Sensitivity / True Positive Rate)", labelpad=10)
    ax.set_ylabel("Precision (PPV / Positive Predictive Value)", labelpad=10)
    
    # Grid lines (light, horizontal and vertical)
    ax.grid(True, linestyle=":", alpha=0.5, color="#888888")
    ax.set_axisbelow(True)
    
    # Set limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Legend
    ax.legend(loc="lower left", framealpha=0.95, facecolor="white", edgecolor="#CCCCCC")
    
    # 4. Save automatically
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    plt.close()
    
    print("\n" + "="*50)
    print("PRECISION-RECALL CURVE GENERATED")
    print("="*50)
    print(f"PNG format: {OUTPUT_PNG}")
    print(f"PDF format: {OUTPUT_PDF}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
