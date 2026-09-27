"""Generate synthetic, anonymised student and course records for LUMEN."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from faker import Faker

MAJORS = ["Computer Science", "Electrical Engineering", "Mechanical Engineering", "Business", "Mathematics"]
DEPARTMENTS = {
    "Computer Science": "CS",
    "Electrical Engineering": "EE",
    "Mechanical Engineering": "ME",
    "Business": "BUS",
    "Mathematics": "MATH",
}

STUDENT_COLUMNS = [
    "student_id",
    "first_name",
    "last_name",
    "email",
    "major",
    "year",
    "gpa",
    "enrollment_date",
    "city",
    "completed_courses",
]

COURSE_CATALOG = [
    ("CS101", "Intro to Programming", "CS", 1, 3, []),
    ("CS201", "Data Structures", "CS", 2, 4, ["CS101"]),
    ("CS301", "Machine Learning", "CS", 3, 4, ["CS201", "MATH201"]),
    ("CS302", "Computer Networks", "CS", 3, 3, ["CS201"]),
    ("EE101", "Circuit Analysis", "EE", 2, 4, []),
    ("EE201", "Digital Systems", "EE", 2, 4, ["EE101"]),
    ("EE301", "Signal Processing", "EE", 3, 4, ["EE201", "MATH201"]),
    ("ME101", "Statics", "ME", 2, 3, []),
    ("ME201", "Thermodynamics", "ME", 3, 4, ["ME101", "MATH201"]),
    ("BUS101", "Principles of Management", "BUS", 1, 3, []),
    ("BUS201", "Financial Accounting", "BUS", 2, 3, ["BUS101"]),
    ("BUS301", "Business Analytics", "BUS", 3, 4, ["BUS201", "CS101"]),
    ("MATH201", "Calculus II", "MATH", 2, 4, []),
    ("MATH301", "Linear Algebra", "MATH", 3, 4, ["MATH201"]),
    ("MATH302", "Probability & Statistics", "MATH", 3, 4, ["MATH201"]),
    ("CS401", "Capstone Software Project", "CS", 4, 6, ["CS301", "CS302"]),
    ("EE401", "Embedded Systems", "EE", 4, 4, ["EE301"]),
    ("BUS401", "Strategic Management", "BUS", 4, 3, ["BUS301"]),
]

UNIVERSITY_EMAIL_DOMAIN = "student.chandigarhuniversity.edu"


def courses_dataframe() -> pd.DataFrame:
    rows = []
    for cid, name, dept, diff, credits, prereqs in COURSE_CATALOG:
        rows.append(
            {
                "course_id": cid,
                "course_name": name,
                "department": dept,
                "difficulty": diff,
                "credits": credits,
                "prerequisites": prereqs,
            }
        )
    return pd.DataFrame(rows)


def students_schema_ok(students_path) -> bool:
    import os

    if not os.path.exists(students_path):
        return False
    header = pd.read_csv(students_path, nrows=0).columns.tolist()
    return all(col in header for col in STUDENT_COLUMNS)


def _major_dept(major: str) -> str:
    return DEPARTMENTS[major]


def _prereqs_met(completed: set[str], prereqs: list[str]) -> bool:
    return all(p in completed for p in prereqs)


def _student_email(fake: Faker, first_name: str, last_name: str) -> str:
    local = f"{first_name.lower()}.{last_name.lower()}".replace("'", "")
    return f"{local}@{UNIVERSITY_EMAIL_DOMAIN}"


def _completed_courses_for_profile(
    major: str,
    year: int,
    course_df: pd.DataFrame,
    course_by_id: dict,
    fake: Faker,
) -> list[str]:
    home = _major_dept(major)
    eligible = course_df[course_df["difficulty"] <= year + 1]
    home_courses = eligible[eligible["department"].isin([home, "MATH"])]

    if len(home_courses) == 0:
        return []

    hi = min(6, len(home_courses) + 1)
    lo = min(2, len(home_courses))
    if hi <= lo:
        n_completed = len(home_courses)
    else:
        n_completed = fake.random_int(min=lo, max=hi - 1)

    picks = home_courses.sample(
        n=min(n_completed, len(home_courses)),
        random_state=fake.random_int(min=0, max=1_000_000),
    )
    completed_ids = picks["course_id"].tolist()
    for cid in completed_ids:
        row = course_by_id[cid]
        for p in row.prerequisites:
            if p not in completed_ids:
                completed_ids.append(p)
    return completed_ids


def _synthetic_fit(
    major: str,
    gpa: float,
    year: int,
    completed: set[str],
    course_dept: str,
    difficulty: int,
    prereqs: list[str],
    rng: np.random.Generator,
) -> float:
    """Rule-based fit score with noise — ground truth for training."""
    score = 0.35
    home = _major_dept(major)
    if course_dept == home:
        score += 0.28
    elif course_dept == "MATH":
        score += 0.12
    elif course_dept == "BUS" and major == "Business":
        score += 0.05

    if _prereqs_met(completed, prereqs):
        score += 0.22
    else:
        score -= 0.35

    gpa_norm = (gpa - 2.0) / 2.0
    diff_gap = difficulty - (year + gpa_norm)
    if diff_gap <= 0:
        score += 0.15
    elif diff_gap == 1:
        score += 0.05
    else:
        score -= 0.12 * diff_gap

    if difficulty >= 4 and gpa < 2.7:
        score -= 0.15

    score += rng.normal(0, 0.06)
    return float(np.clip(score, 0.0, 1.0))


def generate_students(n: int, seed: int = 42) -> pd.DataFrame:
    fake = Faker("en_IN")
    Faker.seed(seed)
    fake.seed_instance(seed)

    course_df = courses_dataframe()
    course_by_id = {r.course_id: r for r in course_df.itertuples()}

    students = []
    for _ in range(n):
        first_name = fake.first_name()
        last_name = fake.last_name()
        major = fake.random_element(MAJORS)
        year = fake.random_int(min=1, max=4)
        gpa = round(fake.pyfloat(min_value=2.0, max_value=4.0), 2)
        enrollment_date = fake.date_between(start_date="-4y", end_date="today").isoformat()
        city = fake.city()
        completed_ids = _completed_courses_for_profile(major, year, course_df, course_by_id, fake)

        students.append(
            {
                "student_id": fake.unique.bothify(text="CU-####??").upper(),
                "first_name": first_name,
                "last_name": last_name,
                "email": _student_email(fake, first_name, last_name),
                "major": major,
                "year": year,
                "gpa": gpa,
                "enrollment_date": enrollment_date,
                "city": city,
                "completed_courses": completed_ids,
            }
        )
    return pd.DataFrame(students, columns=STUDENT_COLUMNS)


def build_training_pairs(students: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    courses = courses_dataframe()
    rows = []
    for stu in students.itertuples():
        completed = set(stu.completed_courses)
        for course in courses.itertuples():
            fit = _synthetic_fit(
                stu.major,
                stu.gpa,
                stu.year,
                completed,
                course.department,
                course.difficulty,
                list(course.prerequisites) if isinstance(course.prerequisites, list) else course.prerequisites,
                rng,
            )
            rows.append(
                {
                    "student_id": stu.student_id,
                    "major": stu.major,
                    "year": stu.year,
                    "gpa": stu.gpa,
                    "completed_courses": stu.completed_courses,
                    "course_id": course.course_id,
                    "course_name": course.course_name,
                    "course_department": course.department,
                    "course_difficulty": course.difficulty,
                    "course_credits": course.credits,
                    "prerequisites": course.prerequisites,
                    "fit_score": fit,
                }
            )
    return pd.DataFrame(rows)


def save_default_dataset(data_dir: str, n_students: int = 200, seed: int = 42) -> None:
    import os

    os.makedirs(data_dir, exist_ok=True)
    students = generate_students(n_students, seed=seed)
    courses = courses_dataframe()
    pairs = build_training_pairs(students, seed=seed)
    students.to_csv(os.path.join(data_dir, "students.csv"), index=False)
    courses.to_csv(os.path.join(data_dir, "courses.csv"), index=False)
    pairs.to_csv(os.path.join(data_dir, "training_pairs.csv"), index=False)


def students_to_json_records(students: pd.DataFrame) -> list[dict]:
    """Export student table as JSON-friendly records (for inspection or APIs)."""
    records = students.copy()
    if "completed_courses" in records.columns:
        records["completed_courses"] = records["completed_courses"].apply(
            lambda x: json.loads(x) if isinstance(x, str) and x.startswith("[") else x
        )
    return records.to_dict(orient="records")
