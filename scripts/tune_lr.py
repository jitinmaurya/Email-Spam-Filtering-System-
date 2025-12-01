# scripts/tune_lr.py
# cspell: disable
import os, json, argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DEFAULT_RS = 42

def load_enron(csv, labelcol="Spam/Ham", subject="Subject", body="Message"):
    df = pd.read_csv(csv)
    y = df[labelcol].astype(str).str.strip().str.lower().map({"ham": 0, "spam": 1})
    if y.isna().any():
        raise ValueError(f"Unrecognized label values in {labelcol}: {df[labelcol].unique()}")
    X = (df.get(subject, "").fillna("").astype(str) + " " +
         df.get(body, "").fillna("").astype(str)).str.strip()
    return X.values, y.values

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/enron.csv")
    ap.add_argument("--labelcol", default="Spam/Ham")
    ap.add_argument("--subjectcol", default="Subject")
    ap.add_argument("--bodycol", default="Message")
    ap.add_argument("--outdir", default="outputs/enron/tuning/lr")
    ap.add_argument("--cv_splits", type=int, default=5)
    ap.add_argument("--n_iter", type=int, default=20)
    ap.add_argument("--random_state", type=int, default=DEFAULT_RS)
    ap.add_argument("--max_train_samples", type=int, default=8000, help="Subsample train for faster tuning (0=use all)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    X, y = load_enron(args.csv, args.labelcol, args.subjectcol, args.bodycol)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.20, random_state=args.random_state, stratify=y
    )

    if args.max_train_samples and len(Xtr) > args.max_train_samples:
        Xtr, _, ytr, _ = train_test_split(
            Xtr, ytr, train_size=args.max_train_samples, random_state=args.random_state, stratify=ytr
        )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            stop_words="english",
            token_pattern=r"(?u)\b\w\w+\b",
            dtype=np.float32
        )),
        ("clf", LogisticRegression(max_iter=2000, solver="liblinear", random_state=args.random_state))
    ])

    param_dist = {
        "tfidf__ngram_range": [(1,1), (1,2)],
        "tfidf__min_df": [3, 5, 10],
        "tfidf__max_df": [0.90, 0.95, 0.98],
        "clf__C": np.logspace(-2, 2, 12),
        "clf__penalty": ["l2"],
        "clf__class_weight": [None, "balanced"],
    }

    cv = StratifiedKFold(n_splits=args.cv_splits, shuffle=True, random_state=args.random_state)

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=args.n_iter,
        scoring="average_precision",
        cv=cv,
        random_state=args.random_state,
        n_jobs=1,                 # safer on Windows
        pre_dispatch="1*n_jobs",
        verbose=1,
    )

    search.fit(Xtr, ytr)

    best = {
        "model": "lr",
        "best_score_PR_AUC_cv_mean": float(search.best_score_),
        "best_params": search.best_params_,
        "n_train_used_for_tuning": int(len(Xtr)),
    }
    print("\n=== BEST (LR) ===")
    print(json.dumps(best, indent=2))

    with open(os.path.join(args.outdir, "best_lr.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    results = pd.DataFrame(search.cv_results_)
    top = results.sort_values("mean_test_score", ascending=False).head(20)
    top.to_csv(os.path.join(args.outdir, "cv_top20_lr.csv"), index=False, encoding="utf-8")

if __name__ == "__main__":
    main()
