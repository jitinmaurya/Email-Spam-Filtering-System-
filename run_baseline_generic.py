# run_baseline_generic.py
# Usage:
#   python run_baseline_generic.py --csv data/enron.csv --model mnb --outdir outputs/enron_mnb
#   python run_baseline_generic.py --csv data/enron.csv --model lr  --outdir outputs/enron_lr
# Supports flexible CSV schemas: ('text' + 'label') OR ('subject','body','label').
# Saves: metrics.json, metrics.txt, preds.csv

import argparse, re, json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report, average_precision_score, roc_auc_score,
    confusion_matrix, f1_score
)

def normalize(t: str) -> str:
    URL = re.compile(r'https?://\S+')
    EMAIL = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')
    NUM = re.compile(r'\b\d+\b')
    t = (t or "").lower()
    t = URL.sub(" <URL> ", t)
    t = EMAIL.sub(" <EMAIL> ", t)
    t = NUM.sub(" <NUM> ", t)
    return re.sub(r"\s+", " ", t).strip()

def load_csv_flexible(path: Path):
    df = pd.read_csv(path)
    lower_map = {c.lower(): c for c in df.columns}

    # detect label column
    label_col = None
    for k in ["label", "labels", "y", "target", "spam"]:
        if k in lower_map:
            label_col = lower_map[k]
            break
    if label_col is None:
        raise ValueError(f"Could not find label column among {list(df.columns)}")

    # detect text
    if "text" in lower_map:
        text = df[lower_map["text"]].astype(str)
    elif "subject" in lower_map and "body" in lower_map:
        text = (df[lower_map["subject"]].astype(str) + " " + df[lower_map["body"]].astype(str))
    elif "message" in lower_map:
        text = df[lower_map["message"]].astype(str)
    elif "content" in lower_map:
        text = df[lower_map["content"]].astype(str)
    else:
        # fallback: concatenate all non-label string columns
        non_label = [c for c in df.columns if c != label_col and df[c].dtype == object]
        if not non_label:
            raise ValueError("Could not infer text column(s). Expected 'text' or 'subject'+'body'.")
        text = df[non_label].astype(str).agg(" ".join, axis=1)

    y = df[label_col].astype(int).to_numpy()
    X = text.map(normalize).tolist()
    return X, y

def build_vectorizer(max_ngram=2, min_df=2, max_df=0.9):
    return TfidfVectorizer(
        ngram_range=(1, max_ngram),
        min_df=min_df,
        max_df=max_df,
        strip_accents="unicode"
    )

def build_model(kind: str):
    if kind == "mnb":
        return MultinomialNB(alpha=0.5)
    if kind == "lr":
        return LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1)
    raise ValueError("model must be 'mnb' or 'lr'")

def to_probs(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    if hasattr(clf, "decision_function"):
        s = clf.decision_function(X)
        return 1.0 / (1.0 + np.exp(-s))
    raise RuntimeError("Classifier has no probability/score method.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV (Enron or SMS)")
    ap.add_argument("--model", choices=["mnb", "lr"], required=True)
    ap.add_argument("--outdir", default="outputs/run")
    ap.add_argument("--max-ngram", type=int, default=2)
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--max-df", type=float, default=0.9)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    X, y = load_csv_flexible(Path(args.csv))
    # 70/15/15 split (train/val/test)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)

    vec = build_vectorizer(args.max_ngram, args.min_df, args.max_df)
    Xtr = vec.fit_transform(X_tr)
    Xva = vec.transform(X_va)
    Xte = vec.transform(X_te)

    clf = build_model(args.model)
    clf.fit(Xtr, y_tr)

    # choose threshold on validation by best F1
    thresholds = np.linspace(0.1, 0.9, 33)
    p_va = to_probs(clf, Xva)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        f1 = f1_score(y_va, (p_va >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t

    # evaluate on test
    p_te = to_probs(clf, Xte)
    y_pred = (p_te >= best_t).astype(int)

    report = classification_report(y_te, y_pred, output_dict=True, digits=4)
    pr_auc = float(average_precision_score(y_te, p_te))
    roc_auc = float(roc_auc_score(y_te, p_te))
    cm = confusion_matrix(y_te, y_pred).tolist()

    # Save metrics + preds
    (outdir / "preds.csv").write_text(
        "proba,label\n" + "\n".join(f"{pp},{ll}" for pp, ll in zip(p_te, y_te)),
        encoding="utf-8"
    )
    metrics = {
        "model": args.model,
        "val_best_threshold": float(best_t),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "classification_report": report,
        "confusion_matrix": cm
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(outdir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Model: {args.model}\nVal-opt threshold: {best_t:.3f}\n")
        f.write(f"PR-AUC: {pr_auc:.4f}\nROC-AUC: {roc_auc:.4f}\n\n")
        f.write(pd.DataFrame(report).to_string())

    print(f"Saved outputs to: {outdir}")
    print(f"Next: python make_plots.py --preds {outdir}/preds.csv --metrics {outdir}/metrics.json --outdir {outdir}")

if __name__ == "__main__":
    main()
