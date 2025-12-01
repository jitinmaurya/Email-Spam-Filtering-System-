# cspell: disable
import argparse, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save-only backend (avoids Tk errors on Windows)
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE


def load_text_y_ids(csv_path: str,
                    labelcol: str | None,
                    textcol: str | None,
                    subjectcol: str | None,
                    bodycol: str | None,
                    idcol: str | None):
    df = pd.read_csv(csv_path)

    # --- label ---
    if labelcol is None:
        for cand in ["label", "labels", "y", "target", "spam", "Spam/Ham"]:
            if cand in df.columns:
                labelcol = cand; break
    if labelcol is None or labelcol not in df.columns:
        raise ValueError(f"No label column. Pass --labelcol. Columns={list(df.columns)}")

    if df[labelcol].dtype == object:
        norm = df[labelcol].astype(str).str.strip().str.lower()
        mapped = norm.map({"ham": 0, "spam": 1, "0": 0, "1": 1})
        if mapped.isna().any():
            raise ValueError(f"Bad label values in '{labelcol}': {df[labelcol].unique()}")
        y = mapped.astype(int).values
    else:
        y = df[labelcol].astype(int).values

    # --- text ---
    X_text = None
    if textcol:
        if textcol not in df.columns:
            raise ValueError(f"--textcol '{textcol}' not in CSV. Columns={list(df.columns)}")
        X_text = df[textcol].fillna("").astype(str)
    else:
        for cand in ["text", "message", "content", "body_text", "Message", "Body"]:
            if cand in df.columns:
                X_text = df[cand].fillna("").astype(str); break
        if X_text is None:
            subj = None; body = None
            if subjectcol:
                if subjectcol not in df.columns:
                    raise ValueError(f"--subjectcol '{subjectcol}' not in CSV.")
                subj = df[subjectcol]
            else:
                for s in ["subject", "Subject"]:
                    if s in df.columns: subj = df[s]; break
            if bodycol:
                if bodycol not in df.columns:
                    raise ValueError(f"--bodycol '{bodycol}' not in CSV.")
                body = df[bodycol]
            else:
                for b in ["body", "Message", "message", "Body"]:
                    if b in df.columns: body = df[b]; break
            if subj is None and body is None:
                raise ValueError("No text columns. Pass --textcol or --subjectcol/--bodycol.")
            if subj is None: subj = pd.Series([""] * len(df))
            if body is None: body = pd.Series([""] * len(df))
            X_text = (subj.fillna("").astype(str) + " " + body.fillna("").astype(str)).str.strip()

    # --- ids (optional) ---
    if idcol and idcol in df.columns:
        ids = df[idcol].astype(str).values
    else:
        ids = np.arange(len(y))  # fallback: row index as id

    return pd.DataFrame({"text": X_text, "label": y, "id": ids, "row_idx": np.arange(len(y))})


def stratified_subsample(df: pd.DataFrame, label_col: str, n: int, random_state: int = 42) -> pd.DataFrame:
    """Return ~n rows preserving class proportions."""
    counts = df[label_col].value_counts()
    props = counts / counts.sum()
    take = (props * n).round().astype(int).clip(lower=1)

    diff = int(n - take.sum())
    if diff != 0:
        residuals = (props * n) - (props * n).round()
        order = residuals.sort_values(ascending=False).index.tolist()
        for cls in order:
            if diff == 0: break
            take[cls] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1

    parts = []
    for cls, k in take.items():
        parts.append(df[df[label_col] == cls].sample(int(k), random_state=random_state))
    return pd.concat(parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/enron.csv", help="Path to CSV")
    ap.add_argument("--labelcol", default=None, help="e.g., 'Spam/Ham'")
    ap.add_argument("--textcol", default=None)
    ap.add_argument("--subjectcol", default=None)
    ap.add_argument("--bodycol", default=None)
    ap.add_argument("--idcol", default=None)  # e.g., 'Message ID'
    ap.add_argument("--max_samples", type=int, default=2000)
    ap.add_argument("--outdir", default="outputs/enron/tsne")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--perplexity", type=float, default=30.0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_text_y_ids(args.csv, args.labelcol, args.textcol, args.subjectcol, args.bodycol, args.idcol)

    n_total = len(df)
    if n_total > args.max_samples:
        df = stratified_subsample(df, "label", args.max_samples, random_state=args.random_state)

    print(f"[t-SNE] Loaded {n_total}; using {len(df)} rows. Label counts: {df['label'].value_counts().sort_index().to_dict()}")

    # TF-IDF → SVD(50) → t-SNE(2)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.98)
    X = tfidf.fit_transform(df["text"].values)

    svd = TruncatedSVD(n_components=50, random_state=args.random_state)
    X50 = svd.fit_transform(X)

    tsne = TSNE(n_components=2, init="pca", learning_rate="auto",
                perplexity=args.perplexity, random_state=args.random_state)
    Z = tsne.fit_transform(X50)

    # Save 2D points for analysis
    pts = pd.DataFrame({
        "id": df["id"].values,
        "row_idx": df["row_idx"].values,
        "label": df["label"].values,
        "x": Z[:, 0],
        "y": Z[:, 1],
    })
    out_csv = os.path.join(args.outdir, "tsne_points.csv")
    pts.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[t-SNE] Saved points: {out_csv}")

    # Save plot
    plt.figure(figsize=(7, 6))
    mask = pts["label"].values == 0
    plt.scatter(pts.loc[mask, "x"], pts.loc[mask, "y"], s=6, alpha=0.6, label="ham (0)")
    plt.scatter(pts.loc[~mask, "x"], pts.loc[~mask, "y"], s=6, alpha=0.6, label="spam (1)")
    plt.legend()
    plt.title("t-SNE of Enron (TF-IDF → SVD → t-SNE)")
    out_png = os.path.join(args.outdir, "tsne_enron.png")
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"[t-SNE] Saved figure: {out_png}")

    # Save metadata
    meta = {
        "csv": os.path.abspath(args.csv),
        "used_rows": int(len(df)),
        "random_state": int(args.random_state),
        "perplexity": float(args.perplexity),
        "tfidf": {"ngram_range": [1, 2], "min_df": 3, "max_df": 0.98},
        "svd_components": 50,
        "cols_used": {
            "labelcol": args.labelcol,
            "textcol": args.textcol,
            "subjectcol": args.subjectcol,
            "bodycol": args.bodycol,
            "idcol": args.idcol
        }
    }
    with open(os.path.join(args.outdir, "tsne_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[t-SNE] Saved meta: {os.path.join(args.outdir, 'tsne_meta.json')}")


if __name__ == "__main__":
    main()
