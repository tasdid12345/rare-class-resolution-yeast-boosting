# Rare-class classification on OpenML Yeast (stand-in for DNSViz error logs)

Question: can a class-balanced, cross-validation-tuned ensemble beat a default random forest on test macro-F1?

Method:
- Stratified 80/20 split with seed 42. The test split is never used for tuning.
- Improved model: soft-voting ensemble of a class-balanced LightGBM (hyperparameters chosen by 5-fold stratified CV macro-F1 on the training split) and a class-balanced random forest. The mixing weight is chosen by out-of-fold CV on the training split. No isotonic calibration.
- Baseline: default scikit-learn RandomForestClassifier.
- Both models are scored once per seed (42, 43, 44) on the test split. Results are in `results.json`; the chart is in `figures/macro_f1.png`.

Run: `pip install -r requirements.txt && python analysis.py`

Caveat: Yeast is a stand-in, so the results do not transfer directly to DNSSEC. The rarest class has very few examples, so rare-class recall is noisy.

## Reference

This project is motivated by the following paper:

> MI Ashiq, O Hureau, C Deccio, T Chung (2025). *Decoding DNSSEC Errors at Scale: An Automated DNSSEC Error Resolution Framework using Insights from DNSViz Logs*. Proceedings of the 2025 ACM. https://dl.acm.org/doi/abs/10.1145/3730567.3764428

Paper author: Prof. Casey Deccio, Brigham Young University (casey@byu.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
