"""
EduPulse - ML training pipeline.

Generates a synthetic dataset of college students, trains a
StandardScaler + RandomForestRegressor pipeline to predict final CGPA,
evaluates it, and exports:

    backend/model.joblib            -> fitted sklearn Pipeline
    backend/students_dataset.csv    -> the synthetic dataset

NOTE: the data is synthetic. Metrics below measure how well the model
recovers the relationships coded into the generator, NOT real-world
predictive accuracy on actual students.

Usage (from the repository root):
    pip install -r backend/requirements.txt
    python backend/train.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"
DATASET_PATH = BASE_DIR / "students_dataset.csv"

N_STUDENTS = 1_200
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Feature order is a contract: the API (Step 2) must send columns in this
# exact order and with these exact names.
FEATURES = [
    "grade_10",
    "grade_12",
    "current_cgpa",
    "attendance",
    "midterm_score",
    "study_hours_weekly",
]
TARGET = "final_cgpa"


# --------------------------------------------------------------------------
# Synthetic data generation
# --------------------------------------------------------------------------
def generate_dataset(n: int = N_STUDENTS, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Create a synthetic student dataset with realistic correlations.

    A latent 'ability' factor drives past and present performance, so the
    features are correlated with each other (as in real cohorts). Final CGPA
    is then built from those features plus noise, with a mild attendance
    penalty below 75% to add a non-linearity the forest can learn.
    """
    rng = np.random.default_rng(seed)

    ability = rng.normal(0.0, 1.0, n)

    # Board marks: correlated through ability; 12th tracks 10th closely.
    grade_10 = 73 + 9.0 * ability + rng.normal(0, 5.0, n)
    grade_10 = np.clip(grade_10, 50, 100)
    grade_12 = 73 + 7.0 * ability + 0.25 * (grade_10 - 73) + rng.normal(0, 4.5, n)
    grade_12 = np.clip(grade_12, 50, 100)

    # Habits: more able / disciplined students study and attend more.
    study_hours = 14 + 3.5 * ability + rng.normal(0, 6.0, n)
    study_hours = np.clip(study_hours, 1, 35)

    attendance = 79 + 6.0 * ability + 0.35 * (study_hours - 14) + rng.normal(0, 9.0, n)
    attendance = np.clip(attendance, 45, 100)

    # Midterm: ability + effort + showing up.
    midterm = (
        60
        + 10.0 * ability
        + 0.55 * (study_hours - 14)
        + 0.20 * (attendance - 79)
        + rng.normal(0, 8.0, n)
    )
    midterm = np.clip(midterm, 25, 100)

    # Current CGPA: accumulated performance so far.
    current_cgpa = (
        6.9
        + 0.85 * ability
        + 0.015 * (attendance - 79)
        + 0.008 * (midterm - 60)
        + rng.normal(0, 0.45, n)
    )
    current_cgpa = np.clip(current_cgpa, 4.0, 10.0)

    # Target: final CGPA. Positive effects from attendance, midterms and
    # study hours, anchored on current CGPA, plus a low-attendance penalty.
    low_attendance_penalty = 0.015 * np.maximum(0, 75 - attendance)
    final_cgpa = (
        0.20
        + 0.55 * current_cgpa
        + 0.018 * midterm
        + 0.012 * attendance
        + 0.020 * study_hours
        + 0.006 * grade_12
        + 0.003 * grade_10
        - low_attendance_penalty
        + rng.normal(0, 0.30, n)
    )
    final_cgpa = np.clip(final_cgpa, 3.0, 10.0)

    df = pd.DataFrame(
        {
            "grade_10": grade_10,
            "grade_12": grade_12,
            "current_cgpa": current_cgpa,
            "attendance": attendance,
            "midterm_score": midterm,
            "study_hours_weekly": study_hours,
            TARGET: final_cgpa,
        }
    ).round(2)

    return df


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=150,
                    max_depth=8,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main() -> None:
    print("Generating synthetic dataset...")
    df = generate_dataset()
    df.to_csv(DATASET_PATH, index=False)
    print(f"  Saved {len(df)} rows -> {DATASET_PATH}")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    print("Training pipeline (StandardScaler + RandomForestRegressor)...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    print("\nEvaluation on held-out test set")
    print(f"  R2 score : {r2:.4f}")
    print(f"  MAE      : {mae:.4f} CGPA points")

    importances = pd.Series(
        pipeline.named_steps["model"].feature_importances_, index=FEATURES
    ).sort_values(ascending=False)
    print("\nGlobal feature importances")
    for name, value in importances.items():
        print(f"  {name:<20} {value:.4f}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel exported -> {MODEL_PATH}")
    print("Note: trained on synthetic data; metrics are not real-world accuracy.")


if __name__ == "__main__":
    main()
