#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate a saved TF-IDF + classifier pipeline on a CSV.
- Loads model.pkl (joblib)
- Reads Subject+Message -> Text, label Spam/Ham -> {ham:0, spam:1}
- Uses threshold: CLI --threshold > metrics.json(threshold_used) > 0.5
- Saves metrics_eval.json, PR/ROC plots, confusion matrices (raw + normalized)

Usage examples:
  python scripts/evaluate_saved_model.py \
    --csv data/enron.csv \
    --model outputs/enron/final/lr/model.pkl \
    --outdir outputs/enron/final/lr_eval

  python scripts/evaluate_saved_model.py ^
    --csv data\enron.csv ^
    --model outputs\enron\final\mnb\model.pkl ^
    --outdir outputs\enron\final\mnb_eval ^
    --threshold 0.62
"""

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

# Headless plotting (no Tk)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, roc_curve, auc,
    average_precision_score, roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score
)
import joblib

def load_enron(csv_path, labelcol="Spam/Ham", subject="Subject", body="Message"):
    df = pd.read_csv(csv_path)
    y = df[labelcol].astype(str).str.strip().str.lower().map({"ham":0, "spam":1})
    if y.isna().any():
        raise ValueError(f"Unrecognized labels in {labelcol}: {df[labelcol].unique()}")
    X = (df.get(subject, "").fillna("").astype(str) + " " +
         df.get(body, "").fillna("").astype(str)).str.strip().values
    return X, y.values

def pick_threshold(cli_threshold, model_path):
    if cli_threshold is not None:
        return float(cli_threshold), "cli"
    # try metrics.json next to model.pkl
    mjson = Path(model_path).with_name("metrics.json")
    if mjson.exists():
        try:
            data = json.loads(mjson.read_text(encoding="utf-8"))
            if "threshold_used" in data:
                return float(data["threshold_used"]), "metrics.json"
        except Exception:
            pass
    return 0.5, "default_0.5"

def plot_pr_roc(y_true, scores, outdir, prefix):
    # PR
    p, r, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    plt.figure()
    plt.plot(r, p, lw=2)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"{prefix} PR curve (AP={ap:.4f})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(outdir, f"{prefix}_pr_curve.png"), dpi=150)
    plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_true, scores)
    rocauc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, lw=2)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"{prefix} ROC curve (AUC={rocauc:.4f})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(outdir, f"{prefix}_roc_curve.png"), dpi=150)
    plt.close()

def save_confusions(y_true, y_pred, outdir, prefix):
    # raw
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Ham (0)","Spam (1)"]); ax.set_yticklabels(["Ham (0)","Spam (1)"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{prefix} Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(Path(outdir, f"{prefix}_cm.png"), dpi=150)
    plt.close()

    # normalized (row-wise)
    cmn = confusion_matrix(y_true, y_pred, labels=[0,1], normalize="true")
    fig, ax = plt.subplots()
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Ham (0)","Spam (1)"]); ax.set_yticklabels(["Ham (0)","Spam (1)"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{prefix} Confusion Matrix (normalized by true)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(Path(outdir, f"{prefix}_cm_norm_true.png"), dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with Subject, Message, Spam/Ham")
    ap.add_argument("--model", required=True, help="Path to model.pkl (joblib)")
    ap.add_argument("--outdir", required=True, help="Directory to write evaluation artifacts")
    ap.add_argument("--threshold", type=float, default=None, help="Decision threshold override")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data & model
    X, y = load_enron(args.csv)
    pipe = joblib.load(args.model)

    # Scores
    if hasattr(pipe, "predict_proba"):
        scores = pipe.predict_proba(X)[:, 1]
    else:
        # fallback: decision_function (scale to 0..1 via min-max)
        if hasattr(pipe, "decision_function"):
            raw = pipe.decision_function(X).astype(float)
            lo, hi = raw.min(), raw.max()
            scores = (raw - lo) / (hi - lo + 1e-12)
        else:
            raise RuntimeError("Model has neither predict_proba nor decision_function.")

    thr, source = pick_threshold(args.threshold, args.model)
    y_pred = (scores >= thr).astype(int)

    # Plots
    plot_pr_roc(y, scores, outdir, prefix="eval")
    save_confusions(y, y_pred, outdir, prefix="eval")

    # Metrics
    TN, FP, FN, TP = confusion_matrix(y, y_pred, labels=[0,1]).ravel()
    metrics = {
        "threshold_source": source,
        "threshold_used": float(thr),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
        "TN": int(TN), "FP": int(FP), "FN": int(FN), "TP": int(TP)
    }
    Path(outdir, "metrics_eval.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n=== EVALUATION DONE ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nArtifacts written to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
