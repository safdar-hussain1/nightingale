"""Disease prediction from anonymized patient features.

A leakage-free, imbalance-aware binary classification pipeline:
resampling and scaling happen inside cross-validation folds, model
selection uses stratified CV on ROC-AUC, and the decision threshold is
tuned on a held-out validation split.
"""

__version__ = "1.0.0"
