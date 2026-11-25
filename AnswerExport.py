"""
Session-only storage + CSV export for the MCQ marker.
No database: results live in a plain Python list and disappear when the server restarts.
"""
import csv
import io

N_Q = 10
MAX_STUDENTS = 33   # class size limit (the answer key is not counted)

# The temporary "database": one dict per student
results = []

# The answer key, e.g. ["B","D","A","C","E","B","B","A","D","C"]
answer_key = []


def set_answer_key(key):
    """Store the answer key detected from the teacher's photo."""
    answer_key.clear()
    answer_key.extend(key)


def grade(answers):
    """Compare student answers with the key. Returns the number correct."""
    return sum(1 for a, k in zip(answers, answer_key) if a == k)


def add_result(name, answers):
    """Grade one student and add them to the session list."""
    if len(results) >= MAX_STUDENTS:
        raise ValueError(f"Class is full: maximum {MAX_STUDENTS} students per session.")
    score = grade(answers)
    results.append({
        "name": name,
        "answers": list(answers),
        "score": score,
        "total": N_Q,
    })
    return len(results) - 1          # index, so the UI can reference this row


def update_answer(student_index, question_number, new_letter):
    """Let the teacher fix a wrongly detected answer, then re-grade."""
    r = results[student_index]
    r["answers"][question_number - 1] = new_letter
    r["score"] = grade(r["answers"])


def to_csv():
    """Build the CSV text. First data row is the answer key for easy reference."""
    out = io.StringIO()
    w = csv.writer(out)

   w.writerow(["Student Number"] + [f"Q{i}" for i in range(1, N_Q + 1)] + ["Score", "Total", "Percentage"])
    w.writerow(["ANSWER KEY"] + answer_key + ["", "", ""])

    for r in results:
        pct = round(100 * r["score"] / r["total"], 1)
        w.writerow([r["name"]] + r["answers"] + [r["score"], r["total"], f"{pct}%"])

    return out.getvalue()


def save_csv(path="class_results.csv"):
    """Write the CSV to disk (handy for testing without the web app)."""
    # utf-8-sig adds a marker so Excel displays names with accents correctly
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        f.write(to_csv())
    return path

def remove_result(student_index):
    results.pop(student_index)


if __name__ == "__main__":
    # Quick self-test
    set_answer_key(list("BDACEBBADC"))
    add_result("Alice Tan", list("BDACEBBADC"))        # perfect
    add_result("Ben Lee", list("BDACEBBADA"))          # 9/10
    i = add_result("Chloe Wong", list("ADACEBBBDC"))   # 8/10
    update_answer(i, 1, "B")                           # teacher fixes Q1 -> 9/10
    print(to_csv())
    save_csv()
