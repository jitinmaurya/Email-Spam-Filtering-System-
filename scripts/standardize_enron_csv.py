from pathlib import Path
import pandas as pd

IN  = Path("data/enron.csv")        # put Jitin's CSV here
OUT = Path("data/enron_std.csv")    # standardized output

df = pd.read_csv(IN)

# Rename columns to a consistent schema
# (fall back safely if casing differs)
cols = {c.lower(): c for c in df.columns}
subject_col = cols.get("subject")
body_col    = cols.get("message") or cols.get("body")
label_col   = cols.get("spam/ham") or cols.get("label") or cols.get("target")

if not subject_col or not body_col or not label_col:
    raise ValueError(f"Expected Subject/Message/Spam/Ham columns; got: {list(df.columns)}")

# Map labels: ham->0, spam->1 (case-insensitive, strips spaces)
lab = (df[label_col].astype(str).str.strip().str.lower()
         .map({"ham": 0, "spam": 1}))
if lab.isna().any():
    raise ValueError("Label column contains values other than 'ham'/'spam'.")

out = pd.DataFrame({
    "subject": df[subject_col].astype(str).fillna(""),
    "body":    df[body_col].astype(str).fillna(""),
    "label":   lab.astype(int)
})

OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)
print(f"Wrote {len(out)} rows -> {OUT}")
