"""SHAP-based plain-language explanations for course recommendations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from lumen.features import FEATURE_NAMES, HUMAN_LABELS, student_course_row

_EXPLAINERS: dict[int, shap.TreeExplainer] = {}


def _get_explainer(model) -> shap.TreeExplainer:
    key = id(model)
    if key not in _EXPLAINERS:
        _EXPLAINERS[key] = shap.TreeExplainer(model)
    return _EXPLAINERS[key]


@dataclass
class ExplanationBullet:
    label: str
    direction: str
    magnitude: float
    detail: str


def _direction(shap_val: float) -> str:
    if shap_val > 0.01:
        return "increases"
    if shap_val < -0.01:
        return "decreases"
    return "neutral"


def _detail_for_feature(name: str, value: float, shap_val: float) -> str:
    label = HUMAN_LABELS.get(name, name)
    if name == "major_course_dept_match" and value >= 0.5:
        return "This course belongs to your major department, which strongly supports the recommendation."
    if name == "prereqs_met":
        if value >= 0.5:
            return "You have completed the required prerequisites."
        return "Missing prerequisites lowers how well this course fits right now."
    if name == "already_completed" and value >= 0.5:
        return "You have already taken this course, so it is not a new recommendation."
    if name == "difficulty_vs_readiness":
        if value <= 0:
            return "The course difficulty aligns with your year and GPA."
        return "The course may be challenging relative to your current readiness."
    if name == "gpa":
        return f"Your GPA ({value:.2f}) influences expected success in harder courses."
    if name.startswith("major_") and value >= 0.5:
        return f"Your profile is anchored in {label.replace('Major: ', '')}."

    dir_word = "raises" if shap_val > 0 else "lowers"
    return f"{label} {dir_word} the fit score based on your profile."


def explain_prediction(
    model,
    student: pd.Series,
    course: pd.Series,
    top_k: int = 5,
) -> tuple[float, list[ExplanationBullet]]:
    X = student_course_row(student, course)
    pred = float(model.predict(X)[0])
    pred = float(np.clip(pred, 0.0, 1.0))

    explainer = _get_explainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    sv = np.asarray(shap_values).reshape(-1)

    bullets: list[ExplanationBullet] = []
    for idx in np.argsort(np.abs(sv))[::-1]:
        if len(bullets) >= top_k:
            break
        fname = FEATURE_NAMES[idx]
        val = float(X.iloc[0, idx])
        s = float(sv[idx])
        if abs(s) < 0.005:
            continue
        bullets.append(
            ExplanationBullet(
                label=HUMAN_LABELS.get(fname, fname),
                direction=_direction(s),
                magnitude=abs(s),
                detail=_detail_for_feature(fname, val, s),
            )
        )

    if not bullets:
        bullets.append(
            ExplanationBullet(
                label="Overall profile",
                direction="neutral",
                magnitude=0.0,
                detail="The model sees a moderate fit based on your combined academic signals.",
            )
        )

    return pred, bullets


def format_summary(course_name: str, score: float, bullets: list[ExplanationBullet]) -> str:
    pct = int(round(score * 100))
    lines = [f"**{course_name}** — fit score **{pct}%**", ""]
    for b in bullets:
        arrow = "↑" if b.direction == "increases" else "↓" if b.direction == "decreases" else "•"
        lines.append(f"- {arrow} **{b.label}**: {b.detail}")
    return "\n".join(lines)
