"""Generate predictions for the unlabelled test set.

Run from the repo root (after training):
    python -m disease_prediction.predict
"""

from __future__ import annotations

import joblib
import pandas as pd

from . import config
from .data import load_test


def predict(test_path=None, model_path=None, output_path=None) -> pd.DataFrame:
    """Score the test set and write patient_id, probability, and label."""
    # Safe: the joblib bundle is produced by this repo's own train step,
    # never downloaded from an untrusted source.
    bundle = joblib.load(model_path or config.MODEL_PATH)
    model, threshold = bundle["model"], bundle["threshold"]

    X_test, patient_ids = load_test(test_path or config.TEST_PATH)
    probs = model.predict_proba(X_test)[:, 1]

    out = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "probability": probs.round(6),
            "prediction": (probs >= threshold).astype(int),
        }
    )
    output_path = output_path or config.PREDICTIONS_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(
        f"Wrote {len(out)} predictions -> {output_path} "
        f"({out['prediction'].sum()} flagged positive at threshold {threshold:.2f})"
    )
    return out


if __name__ == "__main__":
    predict()
