# Rare-class classification on OpenML Yeast (stand-in for DNSViz error logs)

Question: can a class-balanced, cross-validation-tuned ensemble beat a default random forest on test macro-F1?

Method:
- Stratified 80/20 split with seed 42. The test split is never used for tuning.
- Improved model: soft-voting ensemble of a class-balanced LightGBM (hyperparameters chosen by 5-fold stratified CV macro-F1 on the training split) and a class-balanced random forest. The mixing weight is chosen by out-of-fold CV on the training split. No isotonic calibration.
- Baseline: default scikit-learn RandomForestClassifier.
- Both models are scored once per seed (42, 43, 44) on the test split. Results are in `results.json`; the chart is in `figures/macro_f1.png`.

Run: `pip install -r requirements.txt && python analysis.py`

Caveat: Yeast is a stand-in, so the results do not transfer directly to DNSSEC. The rarest class has very few examples, so rare-class recall is noisy.

## Results (latest run)

On the OpenML Yeast stand-in, macro-F1 on the test split was 0.514 for a standard random forest and 0.597 for the class-balanced ensemble (three seeds, one split), though the CV-chosen LightGBM weight was 0.

| Metric | Value |
|---|---|
| baseline_score | 0.5144 |
| improved_score | 0.5972 |
| improvement | 0.08277 |
| baseline_score_std | 0.0673 |
| improved_score_std | 0.005147 |
| baseline_rare_recall | 0.3333 |
| improved_rare_recall | 1 |
| baseline_logloss | 1.695 |
| improved_logloss | 1.283 |
| n_seeds | 3 |
| cv_lgb_weight | 0 |

Run on 2026-10-19. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Yeast is a stand-in: the DNSViz logs are not public, so results do not transfer directly to DNSSEC. Single stratified split with fixed seed; the rarest class has 4 training examples, so rare-class recall is noisy. Seeds vary model randomness only, not the split. Tuning and weight selection used CV on the training split only; the test split was scored once per model per seed. Isotonic calibration was dropped because with very few rare-class examples it suppressed rare-class predictions. Data loading uses openml's ARFF parser because scipy.io.arff in recent SciPy versions fails on bytes input.

## Reference

This project is motivated by the following paper:

> MI Ashiq, O Hureau, C Deccio, T Chung (2025). *Decoding DNSSEC Errors at Scale: An Automated DNSSEC Error Resolution Framework using Insights from DNSViz Logs*. Proceedings of the 2025 ACM. https://dl.acm.org/doi/abs/10.1145/3730567.3764428

Paper author: Prof. Casey Deccio, Brigham Young University (casey@byu.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
