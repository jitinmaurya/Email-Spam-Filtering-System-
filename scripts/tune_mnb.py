#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multinomial Naive Bayes + TF-IDF tuner (leakage-safe).
Saves: model.pkl, metrics.json, threshold_table.csv, cv_top20_mnb.csv,
       pr_curve.png, roc_curve.png, mnb_cm.png, mnb_cm_norm_true.png
"""

import os, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, roc_curve, auc,
    average_precision_score, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
)
import joblib

RANDOM_STATE = 42

def load_enron(csv, labelcol="Spam/Ham", subject="Subject", body="Message"):
    df = pd.read_csv(csv)
    y = df[labelcol].astype(str).str.strip().str.lower().map({"ham":0,"spam":1})
    if y.isna().any():
        raise ValueError(f"Unrecognized labels in {labelcol}: {df[labelcol].unique()}")
    X = (df.get(subject, "").fillna("").astype(str) + " " +
         df.get(body, "").fillna("").astype(str)).str.strip()
    return X.values, y.values

def choose_threshold_by_mode(y_true_val, scores_val, mode="balanced", min_prec=0.99):
    p, r, thr = precision_recall_curve(y_true_val, scores_val)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    table = pd.DataFrame({"threshold": thr, "precision": p[:-1], "recall": r[:-1], "f1": f1})
    if mode == "high_precision":
        hp = table[table["precision"] >= min_prec]
        if len(hp):
            hp = hp.sort_values(["recall", "f1"], ascending=[False, False])
            best = hp.iloc[0]
        else:
            best = table.iloc[table["f1"].argmax()]
    else:
        best = table.iloc[table["f1"].argmax()]
    return float(best["threshold"]), table

def plot_pr_roc(y_true, scores, outdir, prefix):
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
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Greens")
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

    cmn = confusion_matrix(y_true, y_pred, labels=[0,1], normalize="true")
    fig, ax = plt.subplots()
    im = ax.imshow(cmn, cmap="Greens", vmin=0, vmax=1)
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
    ap.add_argument("--csv", default="data/enron.csv")
    ap.add_argument("--outdir", default="outputs/enron/final/mnb")
    ap.add_argument("--max_features", type=int, default=5000)
    ap.add_argument("--random_state", type=int, default=RANDOM_STATE)
    ap.add_argument("--n_iter", type=int, default=30)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--mode", choices=["balanced","high_precision"], default="balanced")
    ap.add_argument("--min_hp_precision", type=float, default=0.99)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    X, y = load_enron(args.csv)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=args.random_state, stratify=y
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=0.125, random_state=args.random_state, stratify=y_tr
    )  # 70/10/20

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=args.max_features,
            stop_words="english",
            token_pattern=r"(?u)\b\w\w+\b",
            dtype=np.float32
        )),
        ("clf", MultinomialNB())
    ])

    param_dist = {
        "tfidf__ngram_range": [(1,1), (1,2)],
        "tfidf__min_df": [3, 5, 10],
        "tfidf__max_df": [0.90, 0.95, 0.98],
        "clf__alpha": np.logspace(-3, 0.5, 12),  # ~0.001 .. ~3.16
        "clf__fit_prior": [True, False],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=args.n_iter,
        scoring="average_precision",
        cv=cv,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        verbose=1,
        refit=True,
    )
    search.fit(X_tr, y_tr)

    cvres = pd.DataFrame(search.cv_results_).sort_values("mean_test_score", ascending=False)
    cvres.head(20).to_csv(outdir / "cv_top20_mnb.csv", index=False, encoding="utf-8")

    best: Pipeline = search.best_estimator_
    va_scores = best.predict_proba(X_va)[:, 1]
    thr, thr_table = choose_threshold_by_mode(y_va, va_scores, mode=args.mode, min_prec=args.min_hp_precision)
    thr_table.to_csv(outdir / "threshold_table.csv", index=False)

    te_scores = best.predict_proba(X_te)[:, 1]
    y_pred = (te_scores >= thr).astype(int)

    plot_pr_roc(y_te, te_scores, outdir, prefix="mnb")
    save_confusions(y_te, y_pred, outdir, prefix="mnb")

    TN, FP, FN, TP = confusion_matrix(y_te, y_pred, labels=[0,1]).ravel()
    metrics = {
        "model": "mnb",
        "mode": args.mode,
        "threshold_used": float(thr),
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "precision": float(precision_score(y_te, y_pred, zero_division=0)),
        "recall": float(recall_score(y_te, y_pred, zero_division=0)),
        "f1": float(f1_score(y_te, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_te, te_scores)),
        "pr_auc": float(average_precision_score(y_te, te_scores)),
        "TN": int(TN), "FP": int(FP), "FN": int(FN), "TP": int(TP),
        "best_params": search.best_params_,
    }
    Path(outdir, "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(best, outdir / "model.pkl")

    print("\n=== MNB DONE ===")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
