#!/usr/bin/env python3
"""
AI Anti-Fraud Service: Dataset Class Distribution Visualization Generator.
Loads the real IEEE-CIS transaction dataset from parquet format,
calculates class imbalance statistics, and renders an IEEE-style academic bar chart.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "ieee_train.parquet"
OUTPUT_DIR = BASE_DIR / "reports" / "thesis_figures"
OUTPUT_PNG = OUTPUT_DIR / "dataset_fraud_distribution.png"
OUTPUT_PDF = OUTPUT_DIR / "dataset_fraud_distribution.pdf"

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
    if not DATA_PATH.exists():
        print(f"[!] Error: Dataset not found at '{DATA_PATH}'.")
        print("    Please ensure data preprocessing has been run first.")
        sys.exit(1)

    print(f"[*] Loading dataset from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    
    # Calculate counts and percentages
    counts = df["isFraud"].value_counts().sort_index()
    total = len(df)
    
    print("\n[+] Dataset Statistics Calculated:")
    for label, count in counts.items():
        pct = (count / total) * 100
        class_name = "Legitimate (Class 0)" if label == 0 else "Fraudulent (Class 1)"
        print(f"    - {class_name}: {count:,} transactions ({pct:.2f}%)")
    print(f"    - Total: {total:,} transactions")

    # Labels and visual configurations
    labels = ["Legitimate\n(Class 0)", "Fraudulent\n(Class 1)"]
    values = [counts[0], counts[1]]
    percentages = [(counts[0] / total) * 100, (counts[1] / total) * 100]

    # Initialize plot
    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=300)
    ax.set_facecolor("white")

    # Bar width and colors
    bar_width = 0.5
    # Grayscale styling: different tones and patterns for monochrome print compatibility
    # Legitimate: Dark grey with diagonal lines (/)
    # Fraudulent: Light/medium grey with cross-hatch (x)
    colors = ["#444444", "#888888"]
    hatches = ["//", "xx"]

    bars = ax.bar(
        labels, 
        values, 
        width=bar_width, 
        color=colors, 
        edgecolor="black", 
        linewidth=1.2
    )

    # Apply hatches
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    # Clean axes spines (IEEE styling: remove top and right borders)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    # Labels and Titles
    ax.set_title("IEEE-CIS Dataset: Class Distribution & Imbalance", pad=15)
    ax.set_ylabel("Transaction Count", labelpad=10)
    
    # Format Y axis with thousands separator
    ax.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)))
    )

    # Add subtle background grid lines (horizontal only)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5, color="#888888")
    ax.set_axisbelow(True)  # Draw grid lines behind the bars

    # Annotate bars with exact count and percentage
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        # Add values on top of the bars
        ax.annotate(
            f"{int(height):,}\n({pct:.2f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),  # 5 points vertical offset
            textcoords="offset points",
            ha="center", 
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="black"
        )

    # Adjust vertical limit to leave room for labels on top of the bars
    ax.set_ylim(0, max(values) * 1.15)

    # Save high-resolution outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"\n[+] Academic figure generated and saved:")
    print(f"    - PNG format: {OUTPUT_PNG}")
    print(f"    - PDF format (vector): {OUTPUT_PDF}")

if __name__ == "__main__":
    main()
