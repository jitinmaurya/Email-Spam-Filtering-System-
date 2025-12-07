# make_plots.py
# Usage: python make_plots.py --preds outputs/run/preds.csv --metrics outputs/run/metrics.json --outdir outputs/run

import argparse, json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, ConfusionMatrixDisplay
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.preds)
    with open(args.metrics, "r", encoding="utf-8") as f:
        m = json.load(f)

    y_true = df["label"].to_numpy()
    y_score = df["proba"].to_numpy()

    # PR curve
    p, r, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(r, p)
    plt.figure()
    plt.plot(r, p)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR Curve (AUC={pr_auc:.3f})")
    plt.grid(True); plt.tight_layout()
    plt.savefig(outdir / "pr_curve.png", dpi=160); plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC Curve (AUC={roc_auc:.3f})")
    plt.grid(True); plt.tight_layout()
    plt.savefig(outdir / "roc_curve.png", dpi=160); plt.close()

    # Confusion Matrix (from saved metrics)
    cm = np.array(m.get("confusion_matrix"))
    if cm.size:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(values_format="d", cmap="Blues")
        plt.title("Confusion Matrix (test)")
        plt.tight_layout()
        plt.savefig(outdir / "confusion_matrix.png", dpi=160); plt.close()

    print(f"Saved figures to {outdir}")

if __name__ == "__main__":
    main()
