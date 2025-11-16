import argparse, pandas as pd, numpy as np, re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score
from sklearn.utils import shuffle

def normalize(t:str)->str:
    URL = re.compile(r'https?://\S+'); EMAIL = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b'); NUM = re.compile(r'\b\d+\b')
    t = (t or "").lower()
    t = URL.sub(" <URL> ", t); t = EMAIL.sub(" <EMAIL> ", t); t = NUM.sub(" <NUM> ", t)
    return re.sub(r"\s+", " ", t).strip()

def load_csv(path):
    df = pd.read_csv(path).fillna({"subject":"","body":""})
    df["text"] = (df["subject"].astype(str)+" "+df["body"].astype(str)).map(normalize)
    return df["text"].tolist(), df["label"].astype(int).to_numpy()

def vec(): return TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9, strip_accents="unicode")

def model(kind):
    if kind=="lr":  return LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
    if kind=="svm": return CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3)
    if kind=="mnb": return MultinomialNB(alpha=0.5)
    raise ValueError("model must be lr/svm/mnb")

def proba(clf, X):
    if hasattr(clf,"predict_proba"): return clf.predict_proba(X)[:,1]
    if hasattr(clf,"decision_function"):
        s = clf.decision_function(X); return 1/(1+np.exp(-s))
    raise RuntimeError("no score method")

def main(csv_path, which):
    X, y = load_csv(csv_path); X, y = shuffle(X, y, random_state=42)
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    Xva, Xte, yva, yte   = train_test_split(Xtmp, ytmp, test_size=0.50, stratify=ytmp, random_state=42)

    v = vec(); Xtr_v = v.fit_transform(Xtr); Xva_v = v.transform(Xva); Xte_v = v.transform(Xte)
    clf = model(which); clf.fit(Xtr_v, ytr)

    from sklearn.metrics import f1_score
    thresholds = np.linspace(0.1,0.9,17)
    p_val = proba(clf, Xva_v)
    best_t, best_f1 = 0.5, -1
    for t in thresholds:
        f1 = f1_score(yva, (p_val>=t).astype(int))
        if f1 > best_f1: best_f1, best_t = f1, t

    p_test = proba(clf, Xte_v); y_pred = (p_test>=best_t).astype(int)
    print(f"Model={which.upper()}  Val-opt threshold={best_t:.2f}")
    print(classification_report(yte, y_pred, digits=3))
    try:
        print("PR-AUC:", round(average_precision_score(yte, p_test), 4))
        print("ROC-AUC:", round(roc_auc_score(yte, p_test), 4))
    except Exception: pass

if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/emails_enron.csv")
    ap.add_argument("--model", choices=["lr","svm","mnb"], required=True)
    args = ap.parse_args(); main(args.csv, args.model)
