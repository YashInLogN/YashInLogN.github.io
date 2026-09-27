"""Feature engineering for (student, course) fit prediction."""

from __future__ import annotations

import ast
from typing import Any

import numpy as np
import pandas as pd

from lumen.data.synthetic import DEPARTMENTS, courses_dataframe

MAJOR_COLS = [f"major_{m.replace(' ', '_')}" for m in DEPARTMENTS.keys()]
FEATURE_NAMES = MAJOR_COLS + [
    "gpa",
    "year",
    "major_course_dept_match",
    "math_course",
    "prereqs_met",
    "prereq_count",
    "completed_in_dept",
    "course_difficulty",
    "course_credits",
    "difficulty_vs_readiness",
    "already_completed",
]

HUMAN_LABELS = {
    "gpa": "Your GPA",
    "year": "Academic year",
    "major_course_dept_match": "Course matches your major",
    "math_course": "Mathematics foundation course",
    "prereqs_met": "Prerequisites satisfied",
    "prereq_count": "Number of prerequisites",
    "completed_in_dept": "Prior courses in this department",
    "course_difficulty": "Course difficulty level",
    "course_credits": "Credit load",
    "difficulty_vs_readiness": "Difficulty vs your readiness",
    "already_completed": "You already completed this course",
}
for m in DEPARTMENTS.keys():
    HUMAN_LABELS[f"major_{m.replace(' ', '_')}"] = f"Major: {m}"


def _parse_completed(val: Any) -> list[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        val = val.strip()
        if val.startswith("["):
            return ast.literal_eval(val)
        return [x.strip() for x in val.split("|") if x.strip()]
    return list(val)


def _prereqs_met(completed: set[str], prereqs: list[str]) -> bool:
    return all(p in completed for p in prereqs)


def row_to_features(
    major: str,
    year: int,
    gpa: float,
    completed_courses: list[str],
    course_id: str,
    course_department: str,
    course_difficulty: int,
    course_credits: int,
    prerequisites: list[str],
) -> dict[str, float]:
    completed = set(completed_courses)
    home = DEPARTMENTS.get(major, "")

    feats: dict[str, float] = {c: 0.0 for c in MAJOR_COLS}
    col = f"major_{major.replace(' ', '_')}"
    if col in feats:
        feats[col] = 1.0

    prereqs = prerequisites or []
    prereqs_ok = _prereqs_met(completed, prereqs)
    course_df = courses_dataframe()
    dept_map = dict(zip(course_df["course_id"], course_df["department"]))
    completed_in_dept = sum(1 for cid in completed if dept_map.get(cid) == course_department)

    readiness = year + (gpa - 2.0) / 2.0
    diff_vs = course_difficulty - readiness

    feats.update(
        {
            "gpa": float(gpa),
            "year": float(year),
            "major_course_dept_match": 1.0 if course_department == home else 0.0,
            "math_course": 1.0 if course_department == "MATH" else 0.0,
            "prereqs_met": 1.0 if prereqs_ok else 0.0,
            "prereq_count": float(len(prereqs)),
            "completed_in_dept": float(completed_in_dept),
            "course_difficulty": float(course_difficulty),
            "course_credits": float(course_credits),
            "difficulty_vs_readiness": float(diff_vs),
            "already_completed": 1.0 if course_id in completed else 0.0,
        }
    )
    return feats


def dataframe_to_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in df.itertuples():
        completed = _parse_completed(r.completed_courses)
        prereqs = _parse_completed(getattr(r, "prerequisites", []))
        rows.append(
            row_to_features(
                r.major,
                int(r.year),
                float(r.gpa),
                completed,
                r.course_id,
                r.course_department,
                int(r.course_difficulty),
                int(r.course_credits),
                prereqs,
            )
        )
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


def student_course_row(
    student: pd.Series,
    course: pd.Series,
) -> pd.DataFrame:
    completed = _parse_completed(student["completed_courses"])
    prereqs = _parse_completed(course["prerequisites"])
    feats = row_to_features(
        student["major"],
        int(student["year"]),
        float(student["gpa"]),
        completed,
        course["course_id"],
        course["department"],
        int(course["difficulty"]),
        int(course["credits"]),
        prereqs,
    )
    return pd.DataFrame([feats], columns=FEATURE_NAMES)
