from datasets import load_dataset
import csv, re
from pathlib import Path

OUT = Path("data/emails_spamassassin.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

def clean(s): return re.sub(r"\s+"," ", (s or "")).strip()

ds = load_dataset("talby/spamassassin")  # columns: text, label (0=ham, 1=spam)

rows = []
for split in ("train","test","validation"):
    if split not in ds: continue
    for r in ds[split]:
        txt = r["text"] or ""
        # crude subject extraction
        m = re.search(r"(^|\n)\s*subject:\s*(.*)", txt, flags=re.IGNORECASE)
        subj = clean(m.group(2) if m else "")
        rows.append([subj, clean(txt), int(r["label"])])

with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["subject","body","label"]); w.writerows(rows)
print(f"Wrote {len(rows)} rows -> {OUT}")
