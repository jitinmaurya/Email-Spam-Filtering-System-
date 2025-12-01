# scripts/metrics_to_tables.py
# Build a table from one or more metrics.json files and export CSV / LaTeX.
# Robust to missing fields; it will include whatever it finds.

import argparse, json, os
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

# Keys we try to extract in this order; missing keys are filled with NaN
PREFERRED_ORDER = [
    "model", "variant", "dataset",
    "accuracy", "macro_f1", "macro_precision", "macro_recall",
    "roc_auc", "pr_auc",
    "spam_precision", "spam_recall", "spam_f1",
    "ham_precision",  "ham_recall",  "ham_f1",
    "tn", "fp", "fn", "tp",
    "threshold", "notes"
]

def load_metrics(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # try to infer a friendly name from the folder structure if missing
    # e.g., outputs/enron/final/mnb/metrics.json -> model=mnb, dataset=enron
    if "model" not in data or data["model"] in (None, "", "unknown"):
        parts = [p for p in path.parts]
        # crude heuristics; safe if it fails
        try:
            # look backwards for model directory
            m = parts[-2]
            data.setdefault("model", m)
        except Exception:
            data.setdefault("model", "unknown")

    # infer dataset from top-level outputs/<dataset>/...
    if "dataset" not in data or not data["dataset"]:
        try:
            idx = parts.index("outputs")
            data.setdefault("dataset", parts[idx+1])
        except Exception:
            data.setdefault("dataset", "unknown")

    # normalize some common alternate keys
    aliases = {
        "f1_macro": "macro_f1",
        "precision_macro": "macro_precision",
        "recall_macro": "macro_recall",
        "auc_roc": "roc_auc",
        "auc_pr": "pr_auc",
        "spam_P": "spam_precision",
        "spam_R": "spam_recall",
        "spam_F1": "spam_f1",
        "ham_P":  "ham_precision",
        "ham_R":  "ham_recall",
        "ham_F1": "ham_f1",
        "TN": "tn", "FP": "fp", "FN": "fn", "TP": "tp"
    }
    for k_old, k_new in aliases.items():
        if k_old in data and k_new not in data:
            data[k_new] = data[k_old]

    # also record source path
    data.setdefault("_source", str(path))

    return data

def make_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    # Gather all keys seen across rows (so we don’t lose anything)
    seen = set()
    for r in rows:
        seen.update(r.keys())
    # Final column ordering: our preferred list first, then any extras alphabetically
    ordered = [k for k in PREFERRED_ORDER if k in seen] + \
              sorted([k for k in seen if k not in set(PREFERRED_ORDER)])
    df = pd.DataFrame(rows)
    df = df.reindex(columns=ordered)
    return df

def to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    # Use pandas’ latex, but wrap in a table environment for IEEE-like output
    body = df.to_latex(index=False, escape=True, longtable=False)
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{body}\n"
        "\\end{table}\n"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", nargs="+", required=True,
                    help="One or more paths to metrics.json files.")
    ap.add_argument("--out", required=True,
                    help="Output directory for CSV/LaTeX.")
    ap.add_argument("--caption", default="Model metrics",
                    help="LaTeX caption.")
    ap.add_argument("--label", default="tab:metrics",
                    help="LaTeX label.")
    ap.add_argument("--latex", action="store_true",
                    help="Also write a LaTeX table file.")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for m in args.metrics:
        p = Path(m)
        if not p.exists():
            print(f"[warn] metrics file not found: {p}")
            continue
        rows.append(load_metrics(p))

    if not rows:
        raise SystemExit("No valid metrics files were loaded.")

    df = make_dataframe(rows)

    # Write CSV
    csv_path = outdir / "metrics_table.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[ok] wrote CSV -> {csv_path}")

    # Optionally write LaTeX
    if args.latex:
        tex = to_latex(df, caption=args.caption, label=args.label)
        tex_path = outdir / "metrics_table.tex"
        tex_path.write_text(tex, encoding="utf-8")
        print(f"[ok] wrote LaTeX -> {tex_path}")

    # Pretty-print to terminal
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print("\n=== METRICS TABLE ===")
        print(df)

if __name__ == "__main__":
    main()


