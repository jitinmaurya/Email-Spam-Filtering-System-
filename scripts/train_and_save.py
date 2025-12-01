# scripts/train_and_save.py
# Train a TF-IDF + (MNB|LR) model, save pipeline, metrics (json/csv/txt),
# preds, PR/ROC curves, confusion matrix, threshold table.

import os, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve, roc_curve,
    confusion_matrix, classification_report, precision_score, recall_score, f1_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = 42

def load_enron(csv, labelcol="Spam/Ham", subject="Subject", body="Message"):
    df = pd.read_csv(csv)
    y = (df[labelcol].astype(str).str.strip().str.lower().map({"ham":0,"spam":1}))
    if y.isna().any():
        raise ValueError(f"Unrecognized labels in {labelcol}: {df[labelcol].unique()}")
    text = (df.get(subject, "").fillna("").astype(str) + " " +
            df.get(body, "").fillna("").astype(str)).str.strip()
    return text.values, y.values

def pick_threshold(y_true, y_score, mode="balanced"):
    # return threshold and metrics for that threshold
    # mode in {"balanced", "high_precision"}  -- spam is positive class (1)
    thresholds = np.linspace(0.1, 0.9, 17)
    best = None
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        p_spam = precision_score(y_true, y_pred, zero_division=0)
        r_spam = recall_score(y_true, y_pred, zero_division=0)
        f1_spam = f1_score(y_true, y_pred, zero_division=0)
        # ham metrics
        y_pred_ham = 1 - y_pred
        y_true_ham = 1 - y_true
        p_ham = precision_score(y_true_ham, y_pred_ham, zero_division=0)
        r_ham = recall_score(y_true_ham, y_pred_ham, zero_division=0)
        f1_ham = f1_score(y_true_ham, y_pred_ham, zero_division=0)

        score_key = f1_spam if mode == "balanced" else (p_spam - 0.002* (1-r_spam))
        cand = dict(threshold=float(t), spam_P=float(p_spam), spam_R=float(r_spam), spam_F1=float(f1_spam),
                    ham_P=float(p_ham), ham_R=float(r_ham), ham_F1=float(f1_ham))
        cand["_score"] = float(score_key)
        if (best is None) or (cand["_score"] > best["_score"]):
            best = cand
    return best

def plot_pr_roc(y_true, y_score, outdir, title=""):
    pr, rc, th = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    # PR
    plt.figure()
    plt.plot(rc, pr, lw=2)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR curve (AP={ap:.3f}) {title}")
    plt.grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig(Path(outdir, "pr_curve.png"), dpi=180)
    plt.close()

    # ROC
    plt.figure()
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0,1],[0,1],"--",alpha=.6)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC curve (AUC={auc:.3f}) {title}")
    plt.grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig(Path(outdir, "roc_curve.png"), dpi=180)
    plt.close()

def plot_confmat(y_true, y_pred, outdir, labels=("ham","spam"), title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i,j], ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(Path(outdir, "confusion_matrix.png"), dpi=180)
    plt.close()
    return cm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/enron.csv")
    ap.add_argument("--outdir", default="outputs/enron/final/mnb")
    ap.add_argument("--model", choices=["mnb","lr"], required=True)
    # TF-IDF
    ap.add_argument("--ngram", default="1-2")
    ap.add_argument("--min_df", type=int, default=5)
    ap.add_argument("--max_df", type=float, default=0.95)
    ap.add_argument("--max_features", type=int, default=30000)
    ap.add_argument("--sublinear_tf", action="store_true", default=True)
    # model params
    ap.add_argument("--alpha", type=float, default=0.3)  # MNB
    ap.add_argument("--C", type=float, default=3.0)      # LR
    # threshold modes
    ap.add_argument("--threshold_mode", choices=["balanced","high_precision"], default="balanced")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    X, y = load_enron(args.csv)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RNG)

    ngram_tuple = tuple(int(k) for k in args.ngram.split("-"))
    vec = TfidfVectorizer(
        ngram_range=ngram_tuple,
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features,
        stop_words="english",
        token_pattern=r"(?u)\b\w\w+\b",
        sublinear_tf=args.sublinear_tf,
        dtype=np.float32
    )

    if args.model == "mnb":
        clf = MultinomialNB(alpha=args.alpha, fit_prior=True)
        model_name = f"MNB(alpha={args.alpha})"
    else:
        clf = LogisticRegression(C=args.C, penalty="l2", solver="liblinear",
                                 max_iter=2000, random_state=RNG)
        model_name = f"LR(C={args.C})"

    pipe = Pipeline([("tfidf", vec), ("clf", clf)])
    pipe.fit(Xtr, ytr)

    # save whole pipeline
    dump(pipe, Path(outdir, "model.pkl"))

    # scores (positive class = spam=1)
    if args.model == "mnb":
        y_score = pipe.predict_proba(Xte)[:,1]
    else:
        # LR: decision_function -> convert to prob-like via sigmoid for plotting
        try:
            y_score = pipe.predict_proba(Xte)[:,1]
        except Exception:
            from scipy.special import expit
            y_score = expit(pipe.decision_function(Xte))

    ap_score = average_precision_score(yte, y_score)
    auc = roc_auc_score(yte, y_score)

    # pick threshold and compute confusion etc.
    best = pick_threshold(yte, y_score, mode=args.threshold_mode)
    thr = best["threshold"]
    y_pred = (y_score >= thr).astype(int)
    cm = confusion_matrix(yte, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()

    # per-class metrics at chosen threshold
    p_spam = best["spam_P"]; r_spam = best["spam_R"]; f1_spam = best["spam_F1"]
    # ham (class 0)
    y_pred_h = 1 - y_pred; y_true_h = 1 - yte
    p_ham = precision_score(y_true_h, y_pred_h, zero_division=0)
    r_ham = recall_score(y_true_h, y_pred_h, zero_division=0)
    f1_ham = f1_score(y_true_h, y_pred_h, zero_division=0)

    # plots
    plot_pr_roc(yte, y_score, outdir, title=model_name)
    plot_confmat(yte, y_pred, outdir, labels=("ham","spam"),
                 title=f"Confusion Matrix @ thr={thr:.2f} ({model_name})")

    # save preds
    pd.DataFrame({"y_true": yte, "y_score": y_score, "y_pred": y_pred}).to_csv(
        Path(outdir, "preds.csv"), index=False
    )

    # metrics
    acc = (tp+tn)/(tp+tn+fp+fn)
    report = classification_report(yte, y_pred, target_names=["ham","spam"], zero_division=0)
    metrics = dict(
        model=args.model, model_name=model_name,
        ngram=args.ngram, min_df=args.min_df, max_df=args.max_df, max_features=args.max_features,
        threshold_mode=args.threshold_mode, threshold_used=float(thr),
        accuracy=float(acc), roc_auc=float(auc), pr_auc=float(ap_score),
        spam_P=float(p_spam), spam_R=float(r_spam), spam_F1=float(f1_spam),
        ham_P=float(p_ham), ham_R=float(r_ham), ham_F1=float(f1_ham),
        TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp)
    )
    # JSON
    Path(outdir, "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # TXT
    with open(Path(outdir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(json.dumps(metrics, indent=2))
        f.write("\n\n==== classification_report ====\n")
        f.write(report)

    # CSV table from the same JSON
    pd.DataFrame([metrics]).to_csv(Path(outdir, "metrics_table.csv"), index=False, encoding="utf-8")

    # simple threshold table around chosen thr
    rows = []
    for t in np.round(np.linspace(max(0, thr-0.15), min(1, thr+0.15), 7), 3):
        yp = (y_score >= t).astype(int)
        cm2 = confusion_matrix(yte, yp, labels=[0,1])
        tn2, fp2, fn2, tp2 = cm2.ravel()
        rows.append(dict(
            thr=float(t),
            spam_P=float(precision_score(yte, yp, zero_division=0)),
            spam_R=float(recall_score(yte, yp, zero_division=0)),
            spam_F1=float(f1_score(yte, yp, zero_division=0)),
            ham_P=float(precision_score(1-yte, 1-yp, zero_division=0)),
            ham_R=float(recall_score(1-yte, 1-yp, zero_division=0)),
            ham_F1=float(f1_score(1-yte, 1-yp, zero_division=0)),
            TN=int(tn2), FP=int(fp2), FN=int(fn2), TP=int(tp2)
        ))
    pd.DataFrame(rows).to_csv(Path(outdir, "threshold_table.csv"), index=False, encoding="utf-8")

if __name__ == "__main__":
    main()
