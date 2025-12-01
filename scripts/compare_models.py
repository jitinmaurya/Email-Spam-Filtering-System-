# scripts/compare_models.py
# Compare baseline vs tuned metrics; robust to missing keys (None).
# cspell: disable
import argparse, json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KEYS = ["accuracy","spam_precision","spam_recall","macro_f1","pr_auc","roc_auc"]

def load_metrics(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d

def safe_float(d, k):
    v = d.get(k, None)
    try:
        return float(v) if v is not None else None
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="metrics.json for baseline")
    ap.add_argument("--tuned", required=True, help="metrics.json for tuned")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    base = load_metrics(args.baseline)
    tuned = load_metrics(args.tuned)

    # Build a tidy table with raw and deltas
    rows = []
    for k in KEYS:
        b = safe_float(base, k)
        t = safe_float(tuned, k)
        delta = (t - b) if (b is not None and t is not None) else None
        rows.append({"metric": k, "baseline": b, "tuned": t, "delta": delta})
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "comparison.csv", index=False, encoding="utf-8")

    # For plotting: coerce missing values to 0.0 so bars render
    plot_df = df.copy()
    plot_df["baseline_plot"] = plot_df["baseline"].fillna(0.0)
    plot_df["tuned_plot"]    = plot_df["tuned"].fillna(0.0)

    # If everything is missing (unlikely), bail gracefully
    if plot_df[["baseline_plot","tuned_plot"]].sum().sum() == 0.0:
        (outdir / "note.txt").write_text(
            "All metrics are None or zero—nothing to plot. Check your inputs.",
            encoding="utf-8"
        )
        return

    x = range(len(plot_df))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.bar([i - w/2 for i in x], plot_df["baseline_plot"], width=w, label="baseline")
    ax.bar([i + w/2 for i in x], plot_df["tuned_plot"],    width=w, label="tuned")
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["metric"], rotation=25, ha="right")
    ax.set_ylabel("score (missing→0)")
    title = f"Compare models | baseline={base.get('model','?')} tuned={tuned.get('model','?')}"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "bars.png", dpi=160)
    plt.close(fig)

    # Also write a short text summary
    lines = []
    lines.append(f"Baseline file: {args.baseline}")
    lines.append(f"Tuned file   : {args.tuned}")
    for _, r in df.iterrows():
        lines.append(f"{r['metric']}: baseline={r['baseline']}  tuned={r['tuned']}  delta={r['delta']}")
    (outdir / "comparison.txt").write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    main()
