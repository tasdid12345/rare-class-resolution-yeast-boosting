# Rare-class error prediction: balanced LightGBM vs random forest

Motivation: DNSSEC misconfigurations are diagnosed from DNSViz error logs, where most error categories are rare and a classifier that ignores them is of little use. The paper "Decoding DNSSEC Errors at Scale: An Automated DNSSEC Error Resolution Framework using Insights from DNSViz Logs" proposes automated error resolution from such logs. This experiment asks a narrower question about the modelling step.

Question: Can a class-balanced, cross-validation-tuned LightGBM classifier beat a standard random forest on macro-F1 for predicting rare error categories in a small imbalanced tabular dataset?

Dataset: the public OpenML Yeast dataset (id 181), fetched with the `openml` Python package. It is a stand-in for the DNSViz error logs, which are not public. It is not the paper's own dataset.

Baseline: a default scikit-learn `RandomForestClassifier`. This is not the paper's method.

Improved model: LightGBM with balanced class weights, a small grid over learning rate, leaves and L2 regularisation chosen by stratified cross-validation on the training split, and isotonic-calibrated probabilities.

Evaluation: one stratified hold-out split with a fixed seed. Macro-F1 on the test split is the main metric; rare-class recall and log-loss are secondary. Randomized models are run over three seeds.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results: written to `results.json` (and `figures/macro_f1.png`).

## Reference

This project is motivated by the following paper:

> MI Ashiq, O Hureau, C Deccio, T Chung (2025). *Decoding DNSSEC Errors at Scale: An Automated DNSSEC Error Resolution Framework using Insights from DNSViz Logs*. Proceedings of the 2025 ACM. https://dl.acm.org/doi/abs/10.1145/3730567.3764428

Paper author: Prof. Casey Deccio, Brigham Young University (casey@byu.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
