# Email Spam Filtering – Classical vs PLMs (COMP 6321)

## Quickstart
```bash
conda env create -f environment.yml
conda activate spamham
jupyter lab
Repo layout

Preprocessing_SVM.ipynb – cleans & vectorizes Enron Spam dataset (TF-IDF).

X_train_tfidf.npz, X_test_tfidf.npz, y_train.npy, y_test.npy – preprocessed artifacts.

tfidf_vectorizer.pkl – fitted vectorizer to use at inference.

notebooks/Visualization.ipynb – EDA on clean text.

notebooks/SVM_Baseline.ipynb – LinearSVC baseline + tuning.

environment.yml – exact dependencies.
Reproducibility

Seeds fixed to 42.

No leakage (fit TF-IDF on train only).
