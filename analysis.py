"""Can a class-balanced ensemble of CV-tuned LightGBM and a class-balanced random forest
beat a default random forest on macro-F1 for rare classes? Stand-in data: OpenML Yeast (id 181).
All tuning uses CV on the training split only; the test split is scored once per model per seed."""
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, log_loss, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import LabelEncoder

SEED = 42
SEEDS = [42, 43, 44]          # randomized models are run with each seed
DATASET_ID = 181
SOURCE_URL = "https://www.openml.org/d/181"
MAX_ROWS = 200_000            # subsample cap to keep runtime small
OUT_DIR = "figures"
ENSEMBLE_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]  # weight on LightGBM vs balanced RF


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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Step 1: tune LightGBM by stratified CV macro-F1 on the training split only.
    print("Tuning LightGBM with stratified CV on the training split...")
    base = lgb.LGBMClassifier(class_weight="balanced", random_state=SEED, verbose=-1)
    grid = {
        "learning_rate": [0.03, 0.1],
        "num_leaves": [7, 31],
        "reg_lambda": [0.0, 5.0],
        "min_child_samples": [5, 20],
    }
    search = GridSearchCV(base, grid, scoring="f1_macro", cv=cv, n_jobs=-1)
    search.fit(X_tr, y_tr)
    best = search.best_params_
    print(f"  best LightGBM params={best}, CV macro-F1={search.best_score_:.4f}")

    # Step 2: choose the ensemble weight by out-of-fold CV predictions on the training split.
    print("Choosing LightGBM/balanced-RF ensemble weight by CV on the training split...")
    lgb_cv = lgb.LGBMClassifier(**best, class_weight="balanced", random_state=SEED, verbose=-1)
    rf_cv = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                   random_state=SEED, n_jobs=-1)
    oof_lgb = cross_val_predict(lgb_cv, X_tr, y_tr, cv=cv, method="predict_proba")
    oof_rf = cross_val_predict(rf_cv, X_tr, y_tr, cv=cv, method="predict_proba")
    cv_scores = {}
    for w in ENSEMBLE_WEIGHTS:
        p = w * oof_lgb + (1 - w) * oof_rf
        cv_scores[w] = f1_score(y_tr, p.argmax(axis=1), average="macro",
                                labels=range(n_classes), zero_division=0)
        print(f"  w_lgb={w:.2f}  CV macro-F1={cv_scores[w]:.4f}")
    w_best = max(ENSEMBLE_WEIGHTS, key=lambda w: (cv_scores[w], -abs(w - 0.5)))
    print(f"  chosen w_lgb={w_best}")

    # Each seed: fit both models on the same training data, score once on the test split.
    scores = {"baseline": [], "improved": []}
    for s in SEEDS:
        print(f"Seed {s}: fitting baseline random forest and improved ensemble...")
        rf_base = RandomForestClassifier(random_state=s).fit(X_tr, y_tr)
        lg = lgb.LGBMClassifier(**best, class_weight="balanced", random_state=s, verbose=-1)
        lg.fit(X_tr, y_tr)
        rf_bal = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                        random_state=s, n_jobs=-1).fit(X_tr, y_tr)
        p_ens = w_best * lg.predict_proba(X_te) + (1 - w_best) * rf_bal.predict_proba(X_te)
        scores["baseline"].append(evaluate(rf_base.predict_proba(X_te), y_te, rare, n_classes))
        scores["improved"].append(evaluate(p_ens, y_te, rare, n_classes))

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
        "cv_lgb_weight": float(w_best),
    }
    print("Metrics:", json.dumps(metrics, indent=2))

    # Optional chart: baseline vs improved macro-F1 with seed std.
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Random forest\n(baseline)", "Balanced ensemble\n(improved)"],
           [b_f1, i_f1], yerr=[b_f1_sd, i_f1_sd], capsize=6, color=["#999999", "#3a7"])
    ax.set_ylabel("Test macro-F1 (mean ± std over seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "macro_f1.png"), dpi=120)
    plt.close(fig)

    results = {
        "question": ("Can a class-balanced, cross-validation-tuned ensemble of LightGBM and a "
                     "balanced random forest beat a standard random forest on macro-F1 for "
                     "predicting rare error categories in a small imbalanced tabular dataset?"),
        "method": ("Stratified 80/20 split (fixed seed); LightGBM grid over learning rate, leaves, "
                   "L2 regularisation and min_child_samples chosen by stratified 5-fold CV on the "
                   "training split; the ensemble weight between LightGBM and a class-balanced "
                   "random forest chosen by out-of-fold CV on the training split; test macro-F1 "
                   "for both models over three seeds."),
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
        "improved": ("Soft-voting ensemble of class-balanced LightGBM (hyperparameters chosen by "
                     "5-fold stratified CV macro-F1 on the training split) and a class-balanced "
                     "random forest (balanced_subsample, 300 trees); mixing weight chosen by "
                     "out-of-fold CV on the training split. No post-hoc probability calibration."),
        "metrics": metrics,
        "notes": (f"Yeast is a stand-in: the DNSViz logs are not public, so results do not "
                  f"transfer directly to DNSSEC. Single stratified split with fixed seed; the "
                  f"rarest class has {int(counts[rare])} training examples, so rare-class "
                  f"recall is noisy. Seeds vary model randomness only, not the split. "
                  f"Tuning and weight selection used CV on the training split only; the test "
                  f"split was scored once per model per seed. Isotonic calibration was dropped "
                  f"because with very few rare-class examples it suppressed rare-class "
                  f"predictions. Data loading uses openml's ARFF parser because scipy.io.arff "
                  f"in recent SciPy versions fails on bytes input."),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Wrote results.json")


if __name__ == "__main__":
    main()
