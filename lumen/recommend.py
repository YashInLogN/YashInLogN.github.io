"""Rank courses for a student using the trained LightGBM model."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from lumen.explain import explain_prediction


def load_model(artifacts_dir: Path):
    payload = joblib.load(artifacts_dir / "lgbm_fit_model.joblib")
    return payload["model"]


def recommend_courses(
    model,
    student: pd.Series,
    courses: pd.DataFrame,
    top_n: int = 5,
    exclude_completed: bool = True,
) -> list[dict]:
    from lumen.features import _parse_completed

    completed = set(_parse_completed(student["completed_courses"]))
    candidates = courses.copy()
    if exclude_completed:
        candidates = candidates[~candidates["course_id"].isin(completed)]

    scored = []
    for course in candidates.itertuples(index=False):
        course_s = pd.Series(course._asdict())
        score, bullets = explain_prediction(model, student, course_s, top_k=5)
        scored.append(
            {
                "course_id": course.course_id,
                "course_name": course.course_name,
                "department": course.department,
                "difficulty": course.difficulty,
                "credits": course.credits,
                "fit_score": score,
                "bullets": bullets,
            }
        )

    scored.sort(key=lambda x: x["fit_score"], reverse=True)
    return scored[:top_n]
