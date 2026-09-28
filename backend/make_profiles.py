"""
Generate SYNTHETIC, anonymised student profiles (the assignment provides no student data).

Each profile is built from the REAL semester spread of its batch, so every course code, title,
credit value and semester is genuine. Only the student, their grades and their situation are invented.
Credits and CGPA are computed with the handbook's own rules so the profiles are internally consistent:
  * Earned credits (clause 8.11): grades O, A+, A, B+, B, C, D or S — never F / FA.
  * CGPA (clause 8.16): sum(credits x grade points) / sum(credits) over graded courses, F/FA count as 0,
    S/U/I grades are excluded.
No real names, IDs or contact details are used — IDs are SYN-xxx.

Run:  python make_profiles.py
"""
import json

from config import PROCESSED_DIR, STUDENT_PROFILES

GRADE_POINTS = {"O": 10, "A+": 9, "A": 8, "B+": 7, "B": 6, "C": 5, "D": 4, "F": 0, "FA": 0}
PASSING = {"O", "A+", "A", "B+", "B", "C", "D", "S"}
CURRENT_SEMESTER = {"2026": 1, "2025": 3, "2024": 5, "2023": 7}  # Sept 2026 = odd term (see term_context.md)

OFFERINGS = json.load(open(PROCESSED_DIR / "semester_offerings.json"))
MINORS = json.load(open(PROCESSED_DIR / "minors.json"))


def courses_for(batch, semesters, minor=None):
    """Real courses a student of `batch` registered for in `semesters` (minor slots filled from minors.json)."""
    out = []
    minor_courses = [m for m in MINORS if m["minor"] == minor and m["batch"] == batch and m["semester"]]
    for o in OFFERINGS:
        if o["batch"] != batch or o["semester"] not in semesters or not o["credits"]:
            continue
        if o["course_code"] == "TBA" and "Minor" in o["title"]:
            picks = [m for m in minor_courses if m["semester"] == o["semester"]]
            for m in picks:
                out.append({"code": m["course_code"], "title": f"{m['title']} ({minor} minor)",
                            "credits": m["credits"], "semester": m["semester"]})
            if not picks:  # open elective with no identifiable code in the source
                out.append({"code": "OPEN-ELECTIVE", "title": o["title"], "credits": o["credits"],
                            "semester": o["semester"]})
            continue
        if o["course_code"] == "TBA":
            continue  # e.g. "Language" / "Elective" slots without a course code
        out.append({"code": o["course_code"].replace(" ", ""), "title": o["title"], "credits": o["credits"],
                    "semester": o["semester"]})
    return out


def build(student_id, batch, done_semesters, default_grade, overrides=None, minor=None, extra=None):
    overrides = overrides or {}
    completed, failed = [], []
    points = graded_credits = earned = 0
    for c in courses_for(batch, done_semesters, minor):
        grade = overrides.get(c["code"], "S" if c["code"] == "DATA801" else default_grade)
        rec = {**c, "grade": grade}
        if grade in GRADE_POINTS:
            points += GRADE_POINTS[grade] * c["credits"]
            graded_credits += c["credits"]
        if grade in PASSING:
            earned += c["credits"]
            completed.append(rec)
        else:
            failed.append({**rec, "attempts": 1})
    profile = {
        "student_id": student_id,
        "programme": f"B.Tech (Data Science curriculum), {batch} batch",
        "batch": batch,
        "current_semester": CURRENT_SEMESTER.get(batch),
        "completed_credits": earned,
        "cgpa": round(points / graded_credits, 2) if graded_credits else None,
        "minor": minor,
        "courses_completed": completed,
        "courses_failed": failed,
    }
    profile.update(extra or {})
    return profile


def main():
    profiles = [
        # Modelled on "Student A" in the brief: passed Linear Algebra & Python, failed Data Structures.
        build("SYN-001", "2024", [1, 2, 3, 4], "B",
              {"COMP201": "F", "MATH301": "F", "MATH204": "B+", "DATA103": "A"},
              extra={"scenario": "Failed Data Structures and Optimization Techniques; wants Machine Learning (needs MATH301)."}),
        # Near graduation, high CGPA -> distinction question.
        build("SYN-002", "2023", [1, 2, 3, 4, 5, 6], "A+", {"COMP202": "A", "COMP208": "A"},
              extra={"scenario": "Final-year student close to graduation, asking about remaining credits and distinction."}),
        # Probation / progression risk: CGPA below 4.00 after year 1 -> cannot register for Semester 3.
        build("SYN-003", "2025", [1, 2], "D",
              {"MATH208": "F", "COMP201": "F", "MATH203": "F", "COMP209": "C"},
              extra={"scenario": "End of year 1 with CGPA below the 4.00 needed to progress to Year 2."}),
        # Missing prerequisite: FA in Data Visualization blocks Advanced EDA (DATA209 needs DATA132 + MATH203).
        build("SYN-004", "2025", [1, 2], "B+", {"DATA132": "FA"},
              extra={"scenario": "Got FA (attendance shortage) in DATA132; wants to take DATA209 Advanced EDA this semester."}),
        # Transfer-credit edge case: claims more external credits than the 40% cap (72 of 180).
        build("SYN-005", "2025", [1, 2], "B",
              extra={"scenario": "Wants to transfer 80 external credits (NPTEL/SWAYAM + another university).",
                     "transfer_credits_requested": {"NPTEL/SWAYAM": 30, "other_university_without_MoU": 50}}),
        # Medical attendance case, used for the handbook-vs-SOP approval conflict.
        build("SYN-006", "2025", [1, 2], "A",
              extra={"scenario": "Hospitalised for 2 weeks this semester; attendance in COMP203 is 68%.",
                     "current_attendance": {"COMP203": 68, "MATH209": 81}}),
        # Deliberately incomplete profile -> advisor must ask a clarifying question.
        {"student_id": "SYN-007", "programme": "B.Tech (Data Science curriculum)", "batch": None,
         "current_semester": None, "completed_credits": None, "cgpa": None, "minor": None,
         "courses_completed": None, "courses_failed": None,
         "scenario": "Incomplete record: batch, semester and course history unknown."},
        # Borderline CGPA for Year-3 progression and for the degree (both need 5.00).
        build("SYN-008", "2024", [1, 2, 3, 4], "C", {"COMP201": "D", "COMP203": "D", "MATH301": "F"},
              extra={"scenario": "CGPA just under the 5.00 needed to progress to Year 3 (Semester 5)."}),
        # Finance minor student (2024 batch minor structure).
        build("SYN-009", "2024", [1, 2, 3, 4], "A", minor="Finance",
              extra={"scenario": "Finance minor; asking which minor course comes next and its prerequisite."}),
        # Psychology minor: prerequisite PSYC101 is not defined anywhere in the supplied data.
        build("SYN-010", "2025", [1, 2], "B+", minor="Psychology",
              extra={"scenario": "Psychology minor; PSYC202 lists PSYC101 as prerequisite, which is not in the data."}),
    ]
    STUDENT_PROFILES.parent.mkdir(parents=True, exist_ok=True)
    STUDENT_PROFILES.write_text(json.dumps(profiles, indent=1))
    for p in profiles:
        print(p["student_id"], p["batch"], "sem", p["current_semester"], "credits", p["completed_credits"],
              "cgpa", p["cgpa"], "failed", [f["code"] + ":" + f["grade"] for f in p["courses_failed"] or []])


if __name__ == "__main__":
    main()
