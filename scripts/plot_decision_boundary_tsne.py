# scripts/plot_decision_boundary_tsne.py
# Visualize a decision boundary by distilling a 2-D surrogate over t-SNE features.

import argparse, joblib
import numpy as np, pandas as pd
from pathlib import Path

from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_enron(csv, labelcol="Spam/Ham", subject="Subject", body="Message"):
    df = pd.read_csv(csv)
    y = df[labelcol].astype(str).str.strip().str.lower().map({"ham":0,"spam":1}).values
    Xtxt = (df.get(subject, "").fillna("").astype(str) + " " +
            df.get(body, "").fillna("").astype(str)).values
    return Xtxt, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/enron.csv")
    ap.add_argument("--model_pkl", required=True)  # pipeline with tfidf + clf
    ap.add_argument("--outdir", default="outputs/enron/tsne_boundary")
    ap.add_argument("--max_samples", type=int, default=2000)
    ap.add_argument("--perplexity", type=float, default=30)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    pipe = joblib.load(args.model_pkl)
    Xtxt, y = load_enron(args.csv)

    if args.max_samples and len(Xtxt) > args.max_samples:
        from sklearn.model_selection import train_test_split
        Xtxt, _, y, _ = train_test_split(
            Xtxt, y, train_size=args.max_samples, stratify=y, random_state=args.random_state
        )

    # 2-D embedding
    Z = pipe.named_steps["tfidf"].transform(Xtxt)  # use vectorizer from pipeline
    emb = TSNE(n_components=2, perplexity=args.perplexity, random_state=args.random_state,
               init="random", learning_rate="auto").fit_transform(Z)

    # teacher predictions from tuned model
    try:
        y_score = pipe.predict_proba(Xtxt)[:,1]
    except Exception:
        from scipy.special import expit
        y_score = expit(pipe.decision_function(Xtxt))

    # fit 2D surrogate on (emb -> y_score)
    surrogate = LogisticRegression(max_iter=2000, solver="lbfgs")
    surrogate.fit(emb, (y_score >= 0.5).astype(int))  # hard labels for boundary
    ap = average_precision_score(y, y_score)

    # grid
    x_min, x_max = emb[:,0].min()-2, emb[:,0].max()+2
    y_min, y_max = emb[:,1].min()-2, emb[:,1].max()+2
    xs = np.linspace(x_min, x_max, 300)
    ys = np.linspace(y_min, y_max, 300)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = surrogate.predict_proba(grid)[:,1].reshape(xx.shape)

    plt.figure(figsize=(7,6))
    plt.contourf(xx, yy, zz, levels=20, alpha=0.25)
    plt.scatter(emb[y==0,0], emb[y==0,1], s=10, label="ham", alpha=0.8)
    plt.scatter(emb[y==1,0], emb[y==1,1], s=10, label="spam", alpha=0.8)
    plt.legend()
    plt.title(f"t-SNE surrogate decision boundary (AP teacher={ap:.3f})")
    plt.tight_layout()
    plt.savefig(outdir / "tsne_boundary.png", dpi=180)
    plt.close()

    pd.DataFrame({"x":emb[:,0], "y":emb[:,1], "y_true":y, "y_score":y_score}).to_csv(
        outdir / "tsne_points.csv", index=False, encoding="utf-8"
    )

if __name__ == "__main__":
    main()
