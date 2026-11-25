"""
10-question, two-column Scantron reader.

Supports dark shading, light shading, blank answers,
multiple answers and uncertain answers requiring review.

Install: python -m pip install opencv-python numpy
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


LETTERS = "ABCDE"
N_QUESTIONS = 10
OPTIONS = 5
WORK_WIDTH = 495


# Strongly shaded bubbles
MAX_SHADED_STD = 27.0
MIN_DARKER_THAN_OTHERS = 12.0

# Lightly shaded bubbles
LIGHT_SHADED_STD = 34.0
LIGHT_DARKNESS = 7.0
LIGHT_MIN_MARGIN = 5.0
LIGHT_STD_ADVANTAGE = 4.0

# Uncertain shading
REVIEW_STD = 42.0
REVIEW_DARKNESS = 5.0

# Possible additional shaded bubbles
SECOND_MARK_STD = 38.0
SECOND_MARK_DARKNESS = 9.0


def find_grid(gray):
    """Return ten rows (Q1-Q5 left, Q6-Q10 right),
    each with five circles."""

    blur = cv2.GaussianBlur(gray, (5, 5), 1)

    found = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=100,
        param2=20,
        minRadius=7,
        maxRadius=14,
    )

    count = 0 if found is None else len(found[0])

    if count != N_QUESTIONS * OPTIONS:
        raise ValueError(
            f"Found {count} circles instead of 50. "
            "Retake a clear, upright photo with the "
            "same framing and all bubbles visible."
        )

    circles = sorted(found[0].tolist(), key=lambda c: c[0])

    columns = [circles[:25], circles[25:]]

    if max(c[0] for c in columns[0]) >= min(
        c[0] for c in columns[1]
    ):
        raise ValueError(
            "Unable to separate the two answer columns."
        )

    rows = []

    for column in columns:

        column.sort(key=lambda c: c[1])

        for j in range(0, 25, OPTIONS):

            row = sorted(
                column[j:j + OPTIONS],
                key=lambda c: c[0]
            )

            if len(row) != OPTIONS:
                raise ValueError(
                    "Missing bubbles in a question row."
                )

            ys = [c[1] for c in row]
            gaps = np.diff([c[0] for c in row])

            if max(ys) - min(ys) > 13:
                raise ValueError(
                    "Bubbles in a question are not "
                    "horizontally aligned."
                )

            if np.any(gaps < 20) or np.any(gaps > 40):
                raise ValueError(
                    "Unexpected spacing between A-E bubbles."
                )

            rows.append(row)

    # Check spacing between question rows
    for offset in (0, 5):

        centers = [
            float(np.median([c[1] for c in row]))
            for row in rows[offset:offset + 5]
        ]

        row_gaps = np.diff(centers)

        if np.any(row_gaps < 55) or np.any(row_gaps > 95):
            raise ValueError(
                "Unexpected spacing between question rows."
            )

    # Check alignment between both columns
    for i in range(5):

        left_y = np.median(
            [c[1] for c in rows[i]]
        )

        right_y = np.median(
            [c[1] for c in rows[i + 5]]
        )

        if abs(left_y - right_y) > 15:
            raise ValueError(
                "Question rows across columns do not line up."
            )

    return rows


def measure(gray, circle):
    """Measure central shading while excluding
    the outer printed bubble ring."""

    x, y, _ = (
        int(round(value)) for value in circle
    )

    radius = 7

    patch = gray[
        y-radius:y+radius+1,
        x-radius:x+radius+1
    ]

    if patch.shape != (15, 15):
        raise ValueError(
            "Bubble too close to photo edge."
        )

    yy, xx = np.ogrid[
        -radius:radius+1,
        -radius:radius+1
    ]

    pixels = patch[xx * xx + yy * yy < 40]

    return (
        float(np.std(pixels)),
        float(np.median(pixels))
    )


def classify_question(stats):
    """
    Classify dark shading, light shading,
    multiple answers, blank answers and uncertain answers.
    """

    stds = [s for s, _ in stats]
    medians = [m for _, m in stats]

    baseline = float(np.median(medians))

    strong = []
    light = []
    possible = []

    # STEP 1: Detect strongly shaded bubbles

    for i in range(OPTIONS):

        others = [
            medians[j]
            for j in range(OPTIONS)
            if j != i
        ]

        darkness = float(
            np.median(others) - medians[i]
        )

        if (
            stds[i] < MAX_SHADED_STD
            and darkness >= MIN_DARKER_THAN_OTHERS
        ):
            strong.append(i)

    # STEP 2: Detect lightly shaded bubbles

    for i in range(OPTIONS):

        if i in strong:
            continue

        others = [
            medians[j]
            for j in range(OPTIONS)
            if j != i
        ]

        other_stds = [
            stds[j]
            for j in range(OPTIONS)
            if j != i
        ]

        darkness = float(
            np.median(others) - medians[i]
        )

        margin = min(others) - medians[i]

        std_advantage = float(
            np.median(other_stds) - stds[i]
        )

        if (
            stds[i] < LIGHT_SHADED_STD
            and darkness >= LIGHT_DARKNESS
            and margin >= LIGHT_MIN_MARGIN
            and std_advantage >= LIGHT_STD_ADVANTAGE
        ):
            light.append(i)

    # STEP 3: Combine confident detections

    marked = sorted(set(strong + light))

    # STEP 4: Detect possible additional marks

    for i in range(OPTIONS):

        if i in marked:
            continue

        darkness = baseline - medians[i]

        if (
            stds[i] < SECOND_MARK_STD
            and darkness >= SECOND_MARK_DARKNESS
        ):
            possible.append(i)

    # STEP 5: Determine the final answer

    if len(marked) >= 2:

        answer = "MULTIPLE"

    elif len(marked) == 1:

        if possible:
            answer = "REVIEW"
        else:
            answer = LETTERS[marked[0]]

    elif possible:

        answer = "REVIEW"

    else:

        darkest = int(np.argmin(medians))

        others = [
            medians[j]
            for j in range(OPTIONS)
            if j != darkest
        ]

        delta = float(
            np.median(others) - medians[darkest]
        )

        # Very faint shading is flagged for review
        # instead of immediately being marked blank.

        if (
            delta >= REVIEW_DARKNESS
            and stds[darkest] < REVIEW_STD
        ):

            answer = "REVIEW"
            possible.append(darkest)

        else:

            answer = "BLANK"

    return answer, marked, possible


def detect_answers(path, debug_path="debug_output_photo.png"):

    img = cv2.imread(str(path))

    if img is None:
        raise ValueError(
            f"Could not open image: {path}"
        )

    height = round(
        img.shape[0] * WORK_WIDTH / img.shape[1]
    )

    img = cv2.resize(
        img,
        (WORK_WIDTH, height),
        interpolation=cv2.INTER_AREA
    )

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    rows = find_grid(gray)

    annotated = img.copy()

    answers = []

    print("Question | Answer   | Std deviations      | Medians")
    print("-" * 76)

    for q, row in enumerate(rows, start=1):

        stats = [
            measure(gray, circle)
            for circle in row
        ]

        answer, marked, possible = classify_question(stats)

        answers.append(answer)

        for i, (x, y, _) in enumerate(row):

            # Green = confidently detected mark
            # Orange = possible mark requiring review
            # Blue = unmarked bubble

            color = (
                (0, 190, 0) if i in marked else
                (0, 165, 255) if i in possible else
                (255, 0, 0)
            )

            cv2.circle(
                annotated,
                (round(x), round(y)),
                13,
                color,
                2
            )

        x, y, _ = row[0]

        cv2.putText(
            annotated,
            f"{q}:{answer}",
            (
                max(0, int(x)-55),
                max(12, int(y)-16)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (0, 0, 210),
            1
        )

        stds = [
            round(s) for s, _ in stats
        ]

        medians = [
            round(m) for _, m in stats
        ]

        print(
            f"Q{q:02d}      | {answer:<8} | "
            f"{str(stds):<19} | {medians}"
        )

    if not cv2.imwrite(str(debug_path), annotated):

        raise ValueError(
            f"Could not save debug image: {debug_path}"
        )

    print("\nANSWERS:", answers)
    print("Debug image:", debug_path)

    return answers


def apply_corrections(answers, entries, label):
    """Manually correct detected answers."""

    updated = answers.copy()

    used = set()

    for entry in entries:

        try:

            question_text, answer = entry.split("=", 1)

            question = int(question_text)

        except ValueError as exc:

            raise ValueError(
                f"Invalid {label} correction '{entry}'; "
                "use 3=B."
            ) from exc

        answer = answer.strip().upper()

        if (
            question not in range(1, N_QUESTIONS + 1)
            or question in used
        ):

            raise ValueError(
                f"Invalid or repeated question number: {question}"
            )

        allowed = (
            set(LETTERS)
            if label == "key"
            else set(LETTERS) | {"BLANK", "MULTIPLE"}
        )

        if answer not in allowed:

            raise ValueError(
                f"Invalid {label} answer '{answer}'. "
                f"Allowed: {sorted(allowed)}"
            )

        used.add(question)

        updated[question - 1] = answer

        print(
            f"Manual correction ({label}): "
            f"Q{question} = {answer}"
        )

    return updated


def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "photo",
        help="Photo of the student Scantron"
    )

    parser.add_argument(
        "--key",
        help="Photo of teacher answer key"
    )

    parser.add_argument(
        "--student-id",
        default="test-student"
    )

    parser.add_argument(
        "--csv",
        default="results.csv"
    )

    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="Q=ANSWER",
        help="Correct a student's detected answer"
    )

    parser.add_argument(
        "--key-set",
        action="append",
        default=[],
        metavar="Q=ANSWER",
        help="Correct a teacher key's detected answer"
    )

    args = parser.parse_args()

    try:

        if args.key_set and not args.key:

            raise ValueError(
                "--key-set requires --key"
            )

        student = detect_answers(args.photo)

        student = apply_corrections(
            student,
            args.set,
            "student"
        )

        if not args.key:

            if args.set:

                print(
                    "Corrected answers:",
                    student
                )

            return

        key = detect_answers(
            args.key,
            "debug_key_photo.png"
        )

        key = apply_corrections(
            key,
            args.key_set,
            "key"
        )

        if any(answer not in LETTERS for answer in key):

            raise ValueError(
                "Teacher key has BLANK/MULTIPLE/REVIEW; "
                "correct with --key-set Q=A"
            )

        if "REVIEW" in student:

            questions = [
                str(i)
                for i, a in enumerate(student, 1)
                if a == "REVIEW"
            ]

            raise ValueError(
                "Student answer(s) need manual review: Q"
                + ", Q".join(questions)
                + ". Correct with --set Q=A or --set Q=BLANK"
            )

        correct = sum(
            s == k
            for s, k in zip(student, key)
        )

        percent = correct * 100 / N_QUESTIONS

        print(
            f"\nGrade: {correct}/{N_QUESTIONS} "
            f"({percent:.0f}%)"
        )

        output = Path(args.csv)

        output.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with output.open(
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            if file.tell() == 0:

                writer.writerow([
                    "Student ID",
                    *[f"Q{i}" for i in range(1, 11)],
                    "Correct",
                    "Percent"
                ])

            student_id = str(args.student_id)

            # Prevent spreadsheet formula interpretation
            if student_id.startswith(
                ("=", "+", "-", "@", "\t", "\r")
            ):

                student_id = "'" + student_id

            writer.writerow([
                student_id,
                *student,
                correct,
                percent
            ])

        print("Saved:", output)

    except (ValueError, OSError, cv2.error) as exc:

        parser.exit(
            1,
            f"ERROR: {exc}\n"
        )


if __name__ == "__main__":
    main()
