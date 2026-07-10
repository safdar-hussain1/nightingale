"""Model candidates and their hyperparameter grids.

Every candidate is a full imblearn ``Pipeline`` so that scaling and SMOTE
resampling are re-fit inside each cross-validation fold — the training
fold is resampled, the scoring fold never is. This is what keeps the
reported scores honest on a 95/5 imbalanced target.
"""

from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import RANDOM_SEED


def build_candidates() -> dict[str, dict]:
    """Return {name: {"pipeline": Pipeline, "params": grid}} for model selection."""
    smote = SMOTE(random_state=RANDOM_SEED)
    return {
        "logistic_regression": {
            "pipeline": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
                ]
            ),
            "params": {"model__C": [0.01, 0.1, 1, 10]},
        },
        "random_forest": {
            "pipeline": Pipeline(
                [
                    ("smote", smote),
                    ("model", RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)),
                ]
            ),
            "params": {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_leaf": [1, 3],
            },
        },
        "gradient_boosting": {
            "pipeline": Pipeline(
                [
                    ("smote", smote),
                    ("model", GradientBoostingClassifier(random_state=RANDOM_SEED)),
                ]
            ),
            "params": {
                "model__n_estimators": [200, 400],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [3, 5],
            },
        },
        "xgboost": {
            "pipeline": Pipeline(
                [
                    (
                        "model",
                        XGBClassifier(
                            random_state=RANDOM_SEED,
                            eval_metric="logloss",
                            # class-weighting instead of SMOTE: positives are ~5%
                            scale_pos_weight=19,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            "params": {
                "model__n_estimators": [200, 400],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [3, 5],
                "model__subsample": [0.8, 1.0],
            },
        },
    }
