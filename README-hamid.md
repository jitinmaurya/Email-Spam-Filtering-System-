# Email-Spam-Filtering-System-
# Hamid – LR/MNB classical baselines

## Scripts
- scripts/tune_lr.py — RandomizedSearchCV over TF–IDF + LR (C, class_weight; ngram_range/min_df/max_df).
- scripts/tune_mnb.py — RandomizedSearchCV over TF–IDF + MNB (alpha, fit_prior; ngram_range/min_df/max_df).
- scripts/metrics_to_tables.py — converts metrics.json into CSV and LaTeX tables (PR/ROC, threshold table).
- scripts/compare_models.py — compares baseline vs tuned metrics and saves a bar chart + summary.

## Outputs (convention)
- outputs/enron/baseline/{lr,mnb}/metrics.json, threshold_table.csv, PR/ROC/CM PNGs
- outputs/enron/final/{lr,mnb}/metrics.json, threshold_table.csv, PR/ROC/CM PNGs
- outputs/enron/compare/ — comparison plots (baseline vs tuned)
- outputs/enron/tables/ — LaTeX table fragments

## Environment
- See `requirements.txt` (or `requirements-lock.txt` to reproduce exactly my environment).
