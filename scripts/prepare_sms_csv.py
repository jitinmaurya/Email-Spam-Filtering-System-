from datasets import load_dataset
import csv, re
from pathlib import Path

OUT = Path("data/emails_sms.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

def clean(s): 
    return re.sub(r"\s+"," ", (s or "")).strip()

# UCI SMS Spam dataset on the Hub (stable)
ds = load_dataset("ucirvine/sms_spam")   # columns: sms, label (0=ham,1=spam)

rows = []
for split in ("train","test","validation"):
    if split not in ds: 
        continue
    for r in ds[split]:
        txt = clean(r["sms"])
        rows.append(["", txt, int(r["label"])])

with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["subject","body","label"])
    w.writerows(rows)

print(f"Wrote {len(rows)} rows -> {OUT}")
