# scripts/prepare_enron_csv.py
import os, csv, re, sys
from pathlib import Path
from email import policy
from email.parser import BytesParser

# Allow passing a custom root:  python scripts/prepare_enron_csv.py data/enron_spam_data
DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/enron_spam_data")
OUT_CSV  = Path("data/emails_enron.csv")
OUT_CSV.parent.mkdir(exist_ok=True, parents=True)

HAM_KEYS  = ("ham", "easy_ham", "hard_ham")
SPAM_KEYS = ("spam",)

def is_ham_dir(p: Path) -> bool:
    name = p.name.lower()
    return any(k in name for k in HAM_KEYS)

def is_spam_dir(p: Path) -> bool:
    name = p.name.lower()
    return any(k in name for k in SPAM_KEYS) and not is_ham_dir(p)  # avoid matching "ham" inside "spam"

def read_text_any(path: Path) -> str:
    """
    Try to parse as RFC822 email; fallback to plain text. Collapse whitespace.
    """
    try:
        raw = path.read_bytes()
    except Exception:
        return ""
    text = ""
    try:
        msg = BytesParser(policy=policy.default).parsebytes(raw)
        # Prefer subject + body if available
        subject = (msg["subject"] or "").strip()
        if msg.is_multipart():
            parts = []
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        parts.append(part.get_content().strip())
                    except Exception:
                        pass
            body = "\n".join(parts)
        else:
            try:
                body = msg.get_content().strip()
            except Exception:
                body = ""
        text = f"Subject: {subject}\n{body}"
    except Exception:
        # Fallback: treat file as plain text
        try:
            text = path.read_text(encoding="latin-1", errors="ignore")
        except Exception:
            text = ""
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_subject(text: str) -> str:
    m = re.search(r"(^|\n)\s*subject:\s*(.*)", text, flags=re.IGNORECASE)
    return m.group(2).strip() if m else ""

rows, n_ham, n_spam = [], 0, 0

if not DATA_DIR.exists():
    raise SystemExit(f"Dataset root not found: {DATA_DIR.resolve()}")

for root, dirs, files in os.walk(DATA_DIR):
    root_p = Path(root)
    label = None
    if is_ham_dir(root_p):
        label = 0; n_ham += 1
    elif is_spam_dir(root_p):
        label = 1; n_spam += 1
    else:
        print("SCAN:", root_p)
        continue

    for fname in files:
        # Skip non-text blobs
        if fname.lower().endswith((".png", ".jpg", ".gif", ".pdf")):
            continue
        fpath = root_p / fname
        txt = read_text_any(fpath)
        if not txt:
            continue
        subj = extract_subject(txt)
        rows.append([subj, txt, label])

print(f"Scanned dirs: ham-like={n_ham}, spam-like={n_spam}")
print(f"Collected messages: {len(rows)}")

with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["subject","body","label"])
    w.writerows(rows)

print(f"Wrote {len(rows)} rows → {OUT_CSV}")
