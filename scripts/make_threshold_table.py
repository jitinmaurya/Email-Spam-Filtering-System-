# scripts/make_threshold_table.py
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def build_table(preds_csv: Path, metrics_json: Path) -> pd.DataFrame:
    df = pd.read_csv(preds_csv)        # expects columns: proba,label
    y  = df["label"].to_numpy()
    p  = df["proba"].to_numpy()

    with open(metrics_json, "r", encoding="utf-8") as f:
        m = json.load(f)
    tstar = float(m.get("val_best_threshold", 0.5))

    # thresholds to report (add t* if not present)
    cands = {0.50, 0.55, 0.60, 0.65, 0.70, round(tstar, 2)}
    rows = []
    for t in sorted(cands):
        yhat = (p >= t).astype(int)
        # class order: [spam(1), ham(0)]
        pr, rc, f1, _ = precision_recall_fscore_support(
            y, yhat, average=None, labels=[1, 0], zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
        rows.append({
            "thr": t,
            "spam_P": pr[0], "spam_R": rc[0], "spam_F1": f1[0],
            "ham_P":  pr[1], "ham_R":  rc[1], "ham_F1":  f1[1],
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "mode": "val-opt" if abs(t - tstar) < 1e-9 else ""
        })
    return pd.DataFrame(rows).sort_values("thr")


def to_markdown(df: pd.DataFrame, title: str = None) -> str:
    md = []
    if title:
        md.append(f"**{title}**")
    md.append("| thr | spam_P | spam_R | spam_F1 | ham_P | ham_R | ham_F1 | TN | FP | FN | TP | mode |")
    md.append("|----:|------:|------:|-------:|-----:|-----:|------:|---:|---:|---:|---:|:----|")
    for _, r in df.iterrows():
        md.append(
            f"| {r.thr:.2f} | {r.spam_P:.3f} | {r.spam_R:.3f} | {r.spam_F1:.3f} | "
            f"{r.ham_P:.3f} | {r.ham_R:.3f} | {r.ham_F1:.3f} | "
            f"{int(r.TN)} | {int(r.FP)} | {int(r.FN)} | {int(r.TP)} | {r.mode} |"
        )
    return "\n".join(md)


def to_latex(df: pd.DataFrame, caption: str = "Threshold table", label: str = "tab:thresholds") -> str:
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lcccccccccc}")
    lines.append(r"\toprule")
    lines.append(r"thr & spam\_P & spam\_R & spam\_F1 & ham\_P & ham\_R & ham\_F1 & TN & FP & FN & TP \\")
    lines.append(r"\midrule")
    for _, r in df.iterrows():
        lines.append(
            f"{r.thr:.2f} & {r.spam_P:.3f} & {r.spam_R:.3f} & {r.spam_F1:.3f} & "
            f"{r.ham_P:.3f} & {r.ham_R:.3f} & {r.ham_F1:.3f} & "
            f"{int(r.TN)} & {int(r.FP)} & {int(r.FN)} & {int(r.TP)} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="e.g., outputs/enron_mnb/preds.csv")
    ap.add_argument("--metrics", required=True, help="e.g., outputs/enron_mnb/metrics.json")
    ap.add_argument("--outdir", required=False, help="folder to save tables; defaults to preds' folder")
    ap.add_argument("--title", default="", help="optional model title for markdown")
    args = ap.parse_args()

    preds = Path(args.preds)
    outdir = Path(args.outdir) if args.outdir else preds.parent
    outdir.mkdir(parents=True, exist_ok=True)

    df = build_table(preds, Path(args.metrics))

    # Save CSV / Markdown / LaTeX
    df.to_csv(outdir / "threshold_table.csv", index=False)
    (outdir / "threshold_table.md").write_text(to_markdown(df, args.title), encoding="utf-8")
    (outdir / "threshold_table.tex").write_text(
        to_latex(df, caption=f"{args.title} thresholds", label="tab:thr"),
        encoding="utf-8"
    )

    print(f"Saved: {outdir/'threshold_table.csv'}")
    print(f"Saved: {outdir/'threshold_table.md'}")
    print(f"Saved: {outdir/'threshold_table.tex'}")


if __name__ == "__main__":
    main()
