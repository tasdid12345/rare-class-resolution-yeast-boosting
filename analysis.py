"""Can a class-balanced, CV-tuned, calibrated LightGBM beat a default random forest
on macro-F1 for rare classes? Stand-in data: OpenML Yeast (id 181).
All tuning uses CV on the training split; the test split is used once per model per seed."""
import json
import os
import sys
import time

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openml
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, log_loss, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

SEED = 42
SEEDS = [42, 43, 44]          # randomized models are run with each seed
DATASET_ID = 181
SOURCE_URL = "https://www.openml.org/d/181"
MAX_ROWS = 200_000            # subsample cap to keep runtime small
OUT_DIR = "figures"


def retry(fn, tries=5):
    """Run fn(); on any failure (incl. HTTP 429/5xx) back off exponentially and retry."""
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"  attempt {i + 1} failed ({type(e).__name__}); retrying")
            time.sleep(2 ** i)


def load_data():
    """Fetch the OpenML dataset as a pandas DataFrame (openml handles ARFF parsing);
    return features, encoded target, row count."""
    print("Fetching dataset from OpenML...")
    meta = retry(lambda: openml.datasets.get_dataset(DATASET_ID, download_data=True))
    target = meta.default_target_attribute
    X_df, y_s, _, _ = retry(
        lambda: meta.get_data(dataset_format="dataframe", target=target))
    df = X_df.copy()
    df[target] = y_s.values
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=SEED)
    X = df.drop(columns=[target]).astype(float).values
    y = LabelEncoder().fit_transform(df[target].astype(str))
    return X, y, len(df)


def evaluate(proba, y_true, rare, n_classes):
    """Main metric: macro-F1. Secondary: recall on the rarest training class, log-loss."""
    pred = proba.argmax(axis=1)
    return {
        "f1": f1_score(y_true, pred, average="macro", labels=range(n_classes), zero_division=0),
        "rare_recall": recall_score(y_true, pred, labels=[rare], average="macro", zero_division=0),
        "logloss": log_loss(y_true, proba, labels=range(n_classes)),
    }


def main():
    X, y, n_rows = load_data()
    n_classes = len(np.unique(y))
    if n_classes < 2:
        sys.exit("Need at least two classes.")

    # One stratified split, fixed seed. Test data is not used for any choice below.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    counts = np.bincount(y_tr)
    rare = int(counts.argmin())  # rarest class in the training split
    print(f"rows={n_rows} train={len(y_tr)} test={len(y_te)} classes={n_classes} "
          f"rarest-class train count={counts[rare]}")

    # Improved model: small grid tuned by stratified CV on the training split only.
    print("Tuning LightGBM with stratified CV on the training split...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    base = lgb.LGBMClassifier(class_weight="balanced", random_state=SEED, verbose=-1)
    grid = {"learning_rate": [0.03, 0.1], "num_leaves": [7, 31], "reg_lambda": [0.0, 5.0]}
    search = GridSearchCV(base, grid, scoring="f1_macro", cv=cv, n_jobs=-1)
    search.fit(X_tr, y_tr)
    best = search.best_params_
    print(f"  best params={best}, CV macro-F1={search.best_score_:.4f}")

    # Each seed: fit both models on the same training data, score once on the test split.
    scores = {"baseline": [], "improved": []}
    for s in SEEDS:
        print(f"Seed {s}: fitting baseline random forest and improved LightGBM...")
        rf = RandomForestClassifier(random_state=s).fit(X_tr, y_tr)
        lg = lgb.LGBMClassifier(**best, class_weight="balanced", random_state=s, verbose=-1)
        cal = CalibratedClassifierCV(lg, method="isotonic", cv=3).fit(X_tr, y_tr)
        scores["baseline"].append(evaluate(rf.predict_proba(X_te), y_te, rare, n_classes))
        scores["improved"].append(evaluate(cal.predict_proba(X_te), y_te, rare, n_classes))

    def agg(name, key):
        """Mean and sample std over seeds."""
        v = np.array([d[key] for d in scores[name]])
        return float(v.mean()), float(v.std(ddof=1))

    b_f1, b_f1_sd = agg("baseline", "f1")
    i_f1, i_f1_sd = agg("improved", "f1")
    if not (np.isfinite(b_f1) and np.isfinite(i_f1)):
        sys.exit("Non-finite metrics; cannot produce a meaningful result.")
    metrics = {
        "baseline_score": b_f1,
        "improved_score": i_f1,
        "improvement": i_f1 - b_f1,  # positive = improved model is better (higher is better)
        "baseline_score_std": b_f1_sd,
        "improved_score_std": i_f1_sd,
        "baseline_rare_recall": agg("baseline", "rare_recall")[0],
        "improved_rare_recall": agg("improved", "rare_recall")[0],
        "baseline_logloss": agg("baseline", "logloss")[0],
        "improved_logloss": agg("improved", "logloss")[0],
        "n_seeds": len(SEEDS),
    }
    print("Metrics:", json.dumps(metrics, indent=2))

    # Optional chart: baseline vs improved macro-F1 with seed std.
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Random forest\n(baseline)", "Balanced LightGBM\n(improved)"],
           [b_f1, i_f1], yerr=[b_f1_sd, i_f1_sd], capsize=6, color=["#999999", "#3a7"])
    ax.set_ylabel("Test macro-F1 (mean ± std over seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "macro_f1.png"), dpi=120)
    plt.close(fig)

    results = {
        "question": ("Can a class-balanced, cross-validation-tuned LightGBM classifier beat a "
                     "standard random forest on macro-F1 for predicting rare error categories "
                     "in a small imbalanced tabular dataset?"),
        "method": ("Stratified 80/20 split (fixed seed); grid over learning rate, leaves and "
                   "L2 regularisation chosen by stratified CV on the training split; "
                   "isotonic-calibrated probabilities; test macro-F1 for both models over "
                   "three seeds."),
        "dataset": {
            "name": "OpenML Yeast (id 181), stand-in for DNSViz error logs",
            "source_url": SOURCE_URL,
            "n_rows": int(n_rows),
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "macro-F1 (test)",
        "higher_is_better": True,
        "baseline": ("Default scikit-learn RandomForestClassifier (random_state=seed, "
                     "all other defaults)."),
        "improved": ("LightGBM with class_weight='balanced'; learning_rate, num_leaves and "
                     "reg_lambda chosen by 5-fold stratified CV macro-F1 on the training split; "
                     "isotonic-calibrated probabilities (internal 3-fold CV)."),
        "metrics": metrics,
        "notes": (f"Yeast is a stand-in: the DNSViz logs are not public, so results do not "
                  f"transfer directly to DNSSEC. Single stratified split with fixed seed; the "
                  f"rarest class has {int(counts[rare])} training examples, so rare-class "
                  f"recall is noisy. Seeds vary model randomness only, not the split. "
                  f"Tuning used CV on the training split only; the test split was scored once "
                  f"per model per seed. Argmax of calibrated probabilities is used for "
                  f"prediction. Data loading uses openml's ARFF parser because scipy.io.arff "
                  f"in recent SciPy versions fails on bytes input."),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Wrote results.json")


if __name__ == "__main__":
    main()
