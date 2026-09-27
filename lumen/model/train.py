"""Train LightGBM course-fit model on synthetic pairs."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from lumen.data.synthetic import save_default_dataset, students_schema_ok
from lumen.features import FEATURE_NAMES, dataframe_to_feature_matrix
from lumen.model.gbm import create_fit_regressor


def load_or_create_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    students_path = data_dir / "students.csv"
    if not students_schema_ok(students_path):
        save_default_dataset(str(data_dir))
    students = pd.read_csv(students_path)
    courses = pd.read_csv(data_dir / "courses.csv")
    pairs = pd.read_csv(data_dir / "training_pairs.csv")
    return students, courses, pairs


def train_model(
    pairs: pd.DataFrame,
    model_path: Path,
    random_state: int = 42,
):
    X = dataframe_to_feature_matrix(pairs)
    y = pairs["fit_score"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    baseline = Ridge(alpha=1.0)
    baseline.fit(X_train, y_train)
    base_pred = baseline.predict(X_test)
    base_rmse = mean_squared_error(y_test, base_pred) ** 0.5
    print(f"Baseline (Ridge) RMSE: {base_rmse:.4f}")

    model, backend = create_fit_regressor(random_state=random_state)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)
    print(f"Backend: {backend}  Hold-out RMSE: {rmse:.4f}  R²: {r2:.4f}")

    os.makedirs(model_path.parent, exist_ok=True)
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES, "backend": backend}, model_path)
    return model


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "data"
    model_path = root / "artifacts" / "lgbm_fit_model.joblib"
    _, _, pairs = load_or_create_data(data_dir)
    train_model(pairs, model_path)


if __name__ == "__main__":
    main()
