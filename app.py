"""LUMEN — Explainable AI Academic Advisor (Streamlit)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from lumen.data.synthetic import (
    MAJORS,
    catalog_guide_dataframe,
    format_course_code,
    generate_students,
    parse_id_list,
)
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


def _profile_value_present(student: pd.Series, key: str) -> bool:
    if key not in student.index:
        return False
    value = student[key]
    return pd.notna(value) and str(value).strip() not in {"", "nan"}


def _format_completed(completed) -> list[str]:
    return [format_course_code(cid) for cid in parse_id_list(completed)]


def render_subject_code_guide() -> None:
    st.markdown(
        "Each recommendation uses a **subject code** (for example `CS101`). "
        "Use this list to see the full subject name, department, and prerequisites."
    )
    query = st.text_input("Search by code, name, or department", placeholder="e.g. CS101 or Machine Learning")
    guide = catalog_guide_dataframe()
    if query.strip():
        needle = query.strip().lower()
        mask = (
            guide["Subject code"].str.lower().str.contains(needle, regex=False)
            | guide["Subject name"].str.lower().str.contains(needle, regex=False)
            | guide["Department"].str.lower().str.contains(needle, regex=False)
        )
        guide = guide.loc[mask]
    st.dataframe(guide, use_container_width=True, hide_index=True, height=560)
    if guide.empty:
        st.info("No subjects match that search.")
    st.caption("Level is the catalog year/difficulty (1 = intro, 4 = capstone).")


@st.dialog("Subject codes", width="large")
def subject_codes_dialog() -> None:
    render_subject_code_guide()


def advisor_page() -> None:
    title_col, btn_col = st.columns([4, 1.5], vertical_alignment="bottom")
    with title_col:
        st.title("LUMEN")
        st.caption("Explainable AI Academic Advisor — recommendations with reasons you can read")
    with btn_col:
        if st.button("View subject codes", type="primary", use_container_width=True):
            subject_codes_dialog()

    students, courses, model = ensure_pipeline()
    st.info("Codes such as **CS101** or **MATH201** are subject IDs. Open **View subject codes** or the **Subject codes** page to see the full names.")

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
            completed = st.multiselect(
                "Completed courses",
                all_ids,
                default=[],
                format_func=format_course_code,
                help="Each code maps to a full subject name. Open Subject codes if you need the list.",
            )
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
        profile = {
            "Major": student["major"],
            "Year": int(student["year"]),
            "GPA": float(student["gpa"]),
            "Completed": _format_completed(student["completed_courses"]),
        }
        if _profile_value_present(student, "first_name"):
            profile = {
                "Name": f"{student['first_name']} {student['last_name']}",
                "Email": student.get("email", ""),
                "Student ID": student["student_id"],
                **profile,
            }
            if _profile_value_present(student, "city"):
                profile["City"] = student["city"]
            if _profile_value_present(student, "enrollment_date"):
                profile["Enrolled"] = student["enrollment_date"]
        st.write(profile)

        top_n = st.slider("Number of recommendations", 3, 8, 5)
        if st.button("Refresh synthetic cohort", help="Regenerate demo students (keeps course catalog)"):
            new_students = generate_students(200, seed=pd.Timestamp.now().microsecond)
            from lumen.data.synthetic import build_training_pairs, courses_dataframe

            pairs = build_training_pairs(new_students)
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            new_students.to_csv(DATA_DIR / "students.csv", index=False)
            courses_dataframe().to_csv(DATA_DIR / "courses.csv", index=False)
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
                    prereqs = rec.get("prerequisites") or []
                    if prereqs:
                        st.caption("Prerequisites: " + ", ".join(format_course_code(p) for p in prereqs))
                    else:
                        st.caption("Prerequisites: none")
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


def subject_codes_page() -> None:
    st.title("Subject codes")
    st.caption("Look up every catalog code used in LUMEN recommendations.")
    render_subject_code_guide()


def main() -> None:
    st.set_page_config(page_title="LUMEN", page_icon="💡", layout="wide")
    page = st.navigation(
        {
            "LUMEN": [
                st.Page(advisor_page, title="Advisor", icon="💡", default=True),
                st.Page(subject_codes_page, title="Subject codes", icon="📘"),
            ]
        }
    )
    page.run()


if __name__ == "__main__":
    main()
