# scripts/threshold_table_to_metrics.py
# Create a metrics.json from a threshold_table.csv by selecting one threshold row.

import argparse, json
from pathlib import Path
import pandas as pd

def pick_row(df: pd.DataFrame, criterion: str):
    """
    criterion options:
      - 'best_spam_f1'       -> max spam_F1
      - 'best_balanced_f1'   -> max (spam_F1 + ham_F1)/2
      - 'threshold=0.60'     -> closest row to a specific threshold
    """
    crit = criterion.strip().lower()
    # normalize expected column names
    cols = {c.lower(): c for c in df.columns}
    def col(*names):
        for n in names:
            if n in cols: return cols[n]
        return None

    thr_col = col("thr", "threshold")
    spf_col = col("spam_f1", "spam_f1 ")
    hmf_col = col("ham_f1", "ham_f1 ")
    if crit.startswith("threshold="):
        if thr_col is None:
            raise SystemExit("No 'thr' or 'threshold' column found.")
        thr = float(crit.split("=",1)[1])
        idx = (df[thr_col] - thr).abs().idxmin()
        return df.loc[idx]
    elif crit == "best_spam_f1":
        if spf_col is None:
            raise SystemExit("No spam_F1 column found.")
        return df.loc[df[spf_col].idxmax()]
    elif crit == "best_balanced_f1":
        if spf_col is None or hmf_col is None:
            raise SystemExit("Need both spam_F1 and ham_F1 for balanced F1.")
        tmp = df.copy()
        tmp["balanced_f1_tmp"] = (tmp[spf_col] + tmp[hmf_col]) / 2.0
        return tmp.loc[tmp["balanced_f1_tmp"].idxmax()]
    else:
        raise SystemExit(f"Unknown criterion: {criterion}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to threshold_table.csv")
    ap.add_argument("--out_metrics", required=True, help="Path to write metrics.json")
    ap.add_argument("--model", required=True, help="Model name (e.g., lr, mnb)")
    ap.add_argument("--dataset", default="enron", help="Dataset name to stamp in JSON")
    ap.add_argument("--criterion", default="best_balanced_f1",
                    help="Selection rule: 'best_spam_f1' | 'best_balanced_f1' | 'threshold=0.60'")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # normalize column names lookup
    cols = {c.lower(): c for c in df.columns}
    def col(*names):
        for n in names:
            if n in cols: return cols[n]
        return None

    tn = col("tn"); fp = col("fp"); fn = col("fn"); tp = col("tp")
    spP = col("spam_p","spam_precision"); spR = col("spam_r","spam_recall"); spF = col("spam_f1")
    hmP = col("ham_p","ham_precision");  hmR = col("ham_r","ham_recall");   hmF = col("ham_f1")
    thr = col("thr","threshold")

    row = pick_row(df, args.criterion)

    # required counts
    for need in (tn, fp, fn, tp):
        if need is None:
            raise SystemExit("Missing one of TN/FP/FN/TP columns in the CSV.")
    TN, FP, FN, TP = int(row[tn]), int(row[fp]), int(row[fn]), int(row[tp])
    total = TN + FP + FN + TP
    accuracy = (TN + TP) / total if total else 0.0

    def get_safe(c):
        return float(row[c]) if c and (c in row) and pd.notna(row[c]) else None

    out = {
        "model": args.model,
        "dataset": args.dataset,
        "threshold": float(row[thr]) if thr and thr in row else None,
        "accuracy": accuracy,
        "spam_precision": get_safe(spP),
        "spam_recall":    get_safe(spR),
        "spam_f1":        get_safe(spF),
        "ham_precision":  get_safe(hmP),
        "ham_recall":     get_safe(hmR),
        "ham_f1":         get_safe(hmF),
        "tn": TN, "fp": FP, "fn": FN, "tp": TP,
        "notes": "Derived from threshold_table.csv",
    }
    if out["spam_f1"] is not None and out["ham_f1"] is not None:
        out["macro_f1"] = (out["spam_f1"] + out["ham_f1"]) / 2.0

    Path(args.out_metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_metrics).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[ok] wrote -> {args.out_metrics}")

if __name__ == "__main__":
    main()
