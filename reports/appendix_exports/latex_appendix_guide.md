# LaTeX Thesis Appendix Integration Guide

This guide provides clean, compile-ready LaTeX templates and parameters to insert the generated anti-fraud backend evaluation plots, API JSON samples, and log traces directly into your Software Engineering bachelor thesis.

---

## 1. LaTeX Package Configuration
Ensure the following packages are declared in your preamble (typically `main.tex` or `preamble.tex`):

```latex
\usepackage{graphicx}      % Required for inserting PNG and PDF plots
\usepackage{listings}      % Required for formatting code and JSON files
\usepackage{xcolor}        % Required for syntax highlighting colors
\usepackage{booktabs}      % Required for academic-quality tables
\usepackage{hyperref}      % For clickable cross-references
```

---

## 2. Listing Setup for JSON and Logs
Define a clean, monospace style for your code, JSON, and log snippets. Add the following to your preamble:

```latex
\definecolor{codegray}{rgb}{0.96,0.96,0.96}
\definecolor{commentgreen}{rgb}{0.13,0.54,0.13}
\definecolor{keywordblue}{rgb}{0.0,0.0,0.8}

\lstdefinestyle{academichighlight}{
    backgroundcolor=\color{codegray},
    commentstyle=\color{commentgreen},
    keywordstyle=\color{keywordblue},
    numberstyle=\tiny\color{gray},
    basicstyle=\ttfamily\footnotesize,
    breakatwhitespace=false,
    breaklines=true,
    captionpos=b,
    keepspaces=true,
    numbers=left,
    numbersep=5pt,
    showspaces=false,
    showstringspaces=false,
    showtabs=false,
    tabsize=2,
    frame=single,
    rulecolor=\color{gray!30}
}
\lstset{style=academichighlight}
```

---

## 3. Inserting Evaluation Plots (Appendix B)

Each vector PDF/PNG figure should be inserted using the standard floating environment. Below are the precise commands, labels, and academic captions.

### A. Precision-Recall Curve
```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.75\textwidth]{reports/thesis_figures/precision_recall_curve.pdf}
    \caption{Sentinel AI: Precision-Recall (PR) Curve evaluated on the validation transaction partition. The dark curve shows the CatBoost fraud detector performance (AP = 0.3953) relative to the random baseline (AP = 0.0344).}
    \label{fig:precision_recall_curve}
\end{figure}
```

### B. Request Latency Distribution
```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.78\textwidth]{reports/thesis_figures/latency_distribution.pdf}
    \caption{Sentinel AI: Request processing latency distribution. Mean execution is recorded at 12.1 ms, and the 95th percentile (P95) is 24.1 ms, demonstrating compliance with the target service level agreement (SLA) limit of 50.0 ms.}
    \label{fig:latency_distribution}
\end{figure}
```

### C. Decision Threshold Matrix
```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.78\textwidth]{reports/thesis_figures/decision_threshold_matrix.pdf}
    \caption{Sentinel AI: Joint decision threshold matrix. Visualizes the classification zones (Allow, Review, Block) mapped against behavioral trust scores (T) and transaction fraud probabilities (P). Scatter points show the validation dataset instances.}
    \label{fig:decision_threshold_matrix}
\end{figure}
```

### D. Global SHAP Feature Influence (Beeswarm)
```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.88\textwidth]{reports/thesis_figures/shap_global_beeswarm.pdf}
    \caption{Sentinel AI: Global model feature influence beeswarm plot. Displays the top 15 features sorted by their cumulative impact on the model's log-odds output. High values of behavioral trust score pull risk down, whereas elevated transaction amounts increase probability.}
    \label{fig:shap_global_beeswarm}
\end{figure}
```

---

## 4. Inserting API Examples and Logs (Appendix A)

Use the `lstinputlisting` command to import the exported files directly from the project directory. This ensures that any update to the backend variables automatically cascades to your thesis PDF when compiled.

### A. Example Fraud Scoring Request
```latex
\subsection{Example Fraud Scoring Request JSON}
The following payload represents an incoming client telemetry record transmitted via the Nginx gateway, containing biometric coordinates, JA3 TLS fingerprints, and transactional values.

\lstinputlisting[
    language=json,
    caption={Biometric-transactional request payload for score-transaction endpoint.},
    label={lst:scoring_request}
]{reports/appendix_exports/appendix_a_scoring_request.json}
```

### B. Example Fraud Scoring Response
```latex
\subsection{Example Fraud Scoring Response JSON}
The API returns the decision verdict along with reason codes, combined probability metrics, and individual sensor risk factors.

\lstinputlisting[
    language=json,
    caption={Decision response from late-fusion engine containing block verdict and trigger reasons.},
    label={lst:scoring_response}
]{reports/appendix_exports/appendix_a_scoring_response.json}
```

### C. Example SHAP Explanation Output
```latex
\subsection{Example SHAP Explanation Output}
The SHAP explanation JSON output maps the local feature importance log-odds contributions for review in the dashboard interface.

\lstinputlisting[
    language=json,
    caption={Model feature contributions computed using TreeExplainer for an anomalous transaction.},
    label={lst:shap_explanation}
]{reports/appendix_exports/appendix_a_shap_explanation.json}
```

### D. Example Backend Production Audit Logs
```latex
\subsection{FastAPI Production Audit Log Sequence}
The log messages are structured in JSON format via the structlog framework to track the request lifecycle and SLA checks.

\lstinputlisting[
    caption={JSON audit log sequence recording transaction processing steps, Redis queries, and CatBoost inference duration.},
    label={lst:backend_logs}
]{reports/appendix_exports/appendix_a_backend_audit.log}
```

---

## 5. Summary Evaluation Metrics Table (Appendix B / Chapter 4)

Below is the academic-grade LaTeX code for the main evaluation metrics table. It uses `booktabs` styling to match standard scientific publishing rules.

```latex
\begin{table}[htbp]
    \centering
    \caption{Sentinel AI: Classification Model Performance Summary}
    \label{tab:evaluation_metrics_summary}
    \begin{tabular}{lr}
        \toprule
        \textbf{Metric Category} & \textbf{Value} \\
        \midrule
        Receiver Operating Characteristic Area (ROC-AUC) & 0.840155 \\
        Precision-Recall Area (PR-AUC) & 0.395318 \\
        Optimal Classification Threshold & 0.446522 \\
        Recall at Selected Threshold (Sensitivity) & 70.00\% \\
        Precision at Selected Threshold (PPV) & 13.76\% \\
        False Positive Rate (FPR) at Selected Threshold & 15.64\% \\
        CatBoost Best Iteration & 1,279 \\
        \midrule
        \textbf{Confusion Matrix Parameters} & \\
        True Negatives (TN) & 96,206 \\
        False Positives (FP) & 17,838 \\
        False Negatives (FN) & 1,219 \\
        True Positives (TP) & 2,845 \\
        \bottomrule
    \end{tabular}
\end{table}
```
