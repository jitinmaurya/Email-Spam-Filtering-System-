# scripts/quick_mnb_sms.py
# Minimal sanity check: TF-IDF (1-2 ngrams) + Multinomial Naive Bayes on SMS dataset

import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score

CSV_PATH = r"data\emails_sms.csv"  # change to forward slashes if you prefer: "data/emails_sms.csv"

def normalize(t: str) -> str:
    """Lowercase + mask URLs/emails/numbers + collapse whitespace (for classic models)."""
    URL = re.compile(r'https?://\S+')
    EMAIL = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')
    NUM = re.compile(r'\b\d+\b')
    t = (t or "").lower()
    t = URL.sub(" <URL> ", t)
    t = EMAIL.sub(" <EMAIL> ", t)
    t = NUM.sub(" <NUM> ", t)
    return re.sub(r"\s+", " ", t).strip()

def main():
    df = pd.read_csv(CSV_PATH).fillna({"subject": "", "body": ""})
    print("Loaded CSV:", df.shape, list(df.columns))
    # Combine subject + body for classic models
    df["text"] = (df["subject"].astype(str) + " " + df["body"].astype(str)).map(normalize)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"].astype(int),
        test_size=0.2, stratify=df["label"], random_state=42
    )

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        strip_accents="unicode"
    )
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = MultinomialNB(alpha=0.5)  # you can try 0.1 and 1.0 too
    clf.fit(Xtr, y_train)

    # Probabilities for metrics
    p = clf.predict_proba(Xte)[:, 1]
    y_pred = (p >= 0.5).astype(int)

    print("\n=== RESULTS (MNB on SMS) ===")
    print(classification_report(y_test, y_pred, digits=3))
    try:
        print("PR-AUC:", round(average_precision_score(y_test, p), 4))
        print("ROC-AUC:", round(roc_auc_score(y_test, p), 4))
    except Exception:
        pass

if __name__ == "__main__":
    main()
