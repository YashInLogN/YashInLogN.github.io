"""LUMEN — Explainable AI Academic Advisor (Streamlit)."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import streamlit as st

from lumen.data.synthetic import MAJORS, courses_dataframe, generate_students
from lumen.explain import format_summary
from lumen.model.train import load_or_create_data, train_model
from lumen.recommend import load_model, recommend_courses

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"


@st.cache_resource
def ensure_pipeline():
    students, courses, pairs = load_or_create_data(DATA_DIR)
    model_path = ARTIFACTS_DIR / "lgbm_fit_model.joblib"
    if not model_path.exists():
        train_model(pairs, model_path)
    model = load_model(ARTIFACTS_DIR)
    return students, courses, model


def main():
    st.set_page_config(page_title="LUMEN", page_icon="💡", layout="wide")
    st.title("LUMEN")
    st.caption("Explainable AI Academic Advisor — recommendations with reasons you can read")

    students, courses, model = ensure_pipeline()

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Student profile")
        mode = st.radio("Profile source", ["Pick from dataset", "Custom profile"], horizontal=True)

        if mode == "Pick from dataset":
            label_by_id = {
                row.student_id: f"{row.first_name} {row.last_name} ({row.student_id})"
                for row in students.itertuples()
            }
            student_ids = students["student_id"].tolist()
            sid = st.selectbox(
                "Student",
                student_ids,
                format_func=lambda x: label_by_id.get(x, x),
            )
            student = students[students["student_id"] == sid].iloc[0].copy()
        else:
            major = st.selectbox("Major", MAJORS)
            year = st.slider("Year", 1, 4, 2)
            gpa = st.slider("GPA", 2.0, 4.0, 3.2, 0.05)
            all_ids = courses["course_id"].tolist()
            completed = st.multiselect("Completed courses", all_ids, default=[])
            student = pd.Series(
                {
                    "student_id": "CUSTOM",
                    "major": major,
                    "year": year,
                    "gpa": gpa,
                    "completed_courses": completed,
                }
            )

        st.markdown("**Current profile**")
        completed_display = student["completed_courses"]
        if isinstance(completed_display, str):
            try:
                completed_display = ast.literal_eval(completed_display)
            except (ValueError, SyntaxError):
                completed_display = completed_display
        profile = {
            "Major": student["major"],
            "Year": int(student["year"]),
            "GPA": float(student["gpa"]),
            "Completed": completed_display,
        }
        if student.get("first_name"):
            profile = {
                "Name": f"{student['first_name']} {student['last_name']}",
                "Email": student.get("email", ""),
                "Student ID": student["student_id"],
                **profile,
            }
            if student.get("city"):
                profile["City"] = student["city"]
            if student.get("enrollment_date"):
                profile["Enrolled"] = student["enrollment_date"]
        st.write(profile)

        top_n = st.slider("Number of recommendations", 3, 8, 5)
        if st.button("Refresh synthetic cohort", help="Regenerate demo students (keeps course catalog)"):
            new_students = generate_students(200, seed=pd.Timestamp.now().microsecond)
            from lumen.data.synthetic import build_training_pairs

            pairs = build_training_pairs(new_students)
            new_students.to_csv(DATA_DIR / "students.csv", index=False)
            pairs.to_csv(DATA_DIR / "training_pairs.csv", index=False)
            train_model(pairs, ARTIFACTS_DIR / "lgbm_fit_model.joblib")
            st.cache_resource.clear()
            st.rerun()

    with col_right:
        st.subheader("Ranked recommendations & explanations")
        recs = recommend_courses(model, student, courses, top_n=top_n)

        if not recs:
            st.info("No courses left to recommend — adjust completed courses or profile.")
            return

        for i, rec in enumerate(recs, start=1):
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"### {i}. {rec['course_name']} (`{rec['course_id']}`)")
                    st.write(
                        f"Department **{rec['department']}** · "
                        f"Difficulty **{rec['difficulty']}** · "
                        f"Credits **{rec['credits']}**"
                    )
                with c2:
                    st.metric("Fit score", f"{rec['fit_score'] * 100:.0f}%")

                st.markdown("**Why this course?**")
                st.markdown(format_summary(rec["course_name"], rec["fit_score"], rec["bullets"]))

    st.divider()
    st.markdown(
        """
        **Stack:** Python · pandas · scikit-learn · LightGBM · SHAP · Streamlit  
        **Data:** Local synthetic records only — no real student data.
        """
    )


if __name__ == "__main__":
    main()
