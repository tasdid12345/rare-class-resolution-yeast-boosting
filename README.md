# Rare class resolution: balanced LightGBM vs random forest (Yeast stand-in)

Question: can a class-balanced, CV-tuned, isotonic-calibrated LightGBM beat a default
random forest on test macro-F1 for rare classes?

Dataset: OpenML Yeast (id 181), used as a stand-in for DNSViz error logs (not public).

Method (unchanged): one stratified 80/20 split with seed 42; LightGBM grid
(learning rate, num_leaves, reg_lambda) tuned by 5-fold stratified CV on the training
split only; isotonic calibration; both models scored once per seed on the test split
over seeds 42, 43, 44.

Run: `python analysis.py` (writes `results.json` and `figures/macro_f1.png`).

Fix note: data loading now goes through the openml package's ARFF parser instead of
`scipy.io.arff.loadarff` on raw bytes, which fails on recent SciPy. The dataset, split,
models and metrics are unchanged.

## Reference

This project is motivated by the following paper:

> MI Ashiq, O Hureau, C Deccio, T Chung (2025). *Decoding DNSSEC Errors at Scale: An Automated DNSSEC Error Resolution Framework using Insights from DNSViz Logs*. Proceedings of the 2025 ACM. https://dl.acm.org/doi/abs/10.1145/3730567.3764428

Paper author: Prof. Casey Deccio, Brigham Young University (casey@byu.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
