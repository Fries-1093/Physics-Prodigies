"""
10-question, two-column Scantron reader.

Supports:
- Heavily shaded bubbles
- Lightly shaded bubbles
- Uneven lighting and shadows
- Blank answers
- Multiple answers
- Uncertain answers requiring review

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


def find_grid(gray):
    """Return ten rows, each containing five circles."""

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
    """
    Measure bubble shading relative to the surrounding paper.

    Local brightness normalization reduces the effect
    of uneven lighting, shadows and bright areas.
    """

    x, y, r = (
        int(round(value)) for value in circle
    )

    radius = max(12, int(r * 1.6))

    patch = gray[
        y-radius:y+radius+1,
        x-radius:x+radius+1
    ]

    if patch.shape != (
        2 * radius + 1,
        2 * radius + 1
    ):
        raise ValueError(
            "Bubble too close to photo edge."
        )

    yy, xx = np.ogrid[
        -radius:radius+1,
        -radius:radius+1
    ]

    distance = xx * xx + yy * yy

    # Measure the interior of the bubble,
    # excluding the outer printed circle.

    inside = distance <= (r * 0.65) ** 2

    pixels = patch[inside]

    # Measure the paper surrounding the bubble.
    # Use areas above and below to avoid nearby options.

    outside = (
        (distance >= (r * 1.15) ** 2)
        & (distance <= (r * 1.5) ** 2)
        & (np.abs(yy) >= r)
    )

    background = patch[outside]

    if len(background) == 0:
        raise ValueError(
            "Unable to measure background."
        )

    # Determine the local paper brightness.

    paper_brightness = float(
        np.median(background)
    )

    # The 85th percentile reduces interference
    # from printed letters inside the bubbles.

    bubble_brightness = float(
        np.percentile(pixels, 85)
    )

    # Higher darkness means more pencil shading.

    darkness = paper_brightness - bubble_brightness

    return (
        float(np.std(pixels)),
        darkness
    )


def classify_question(stats):
    """
    Detect heavy shading, light shading,
    multiple answers, blank answers and uncertain answers.
    """

    stds = [s for s, _ in stats]

    darkness = [d for _, d in stats]

    marked = []
    possible = []

    # STEP 1: Identify shaded bubbles.

    for i in range(OPTIONS):

        # Clearly shaded bubble.

        if (
            darkness[i] >= 30.0
            and stds[i] < 35.0
        ):

            marked.append(i)

        # Lightly shaded bubble.

        elif (
            darkness[i] >= 18.0
            and stds[i] < 28.0
        ):

            marked.append(i)

        # Very faint shading requiring review.

        elif (
            darkness[i] >= 10.0
            and stds[i] < 25.0
        ):

            possible.append(i)

    # STEP 2: Multiple confidently shaded bubbles.

    if len(marked) >= 2:

        return "MULTIPLE", marked, possible

    # STEP 3: One confidently shaded bubble.

    if len(marked) == 1:

        if possible:

            return "REVIEW", marked, possible

        return LETTERS[marked[0]], marked, []

    # STEP 4: Uncertain shading.

    if possible:

        return "REVIEW", [], possible

    # STEP 5: No convincing evidence of shading.

    return "BLANK", [], []


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

    print(
        "Question | Answer   | Std deviations"
        "            | Normalized Darkness"
    )

    print("-" * 90)

    for q, row in enumerate(rows, start=1):

        stats = [
            measure(gray, circle)
            for circle in row
        ]

        answer, marked, possible = classify_question(stats)

        answers.append(answer)

        for i, (x, y, _) in enumerate(row):

            # Green = confidently shaded.
            # Orange = possible shading.
            # Blue = unmarked.

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
            round(s, 1) for s, _ in stats
        ]

        darkness = [
            round(d, 1) for _, d in stats
        ]

        print(
            f"Q{q:02d}      | {answer:<8} | "
            f"{str(stds):<25} | {darkness}"
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
                    "Student Number",
                    *[f"Q{i}" for i in range(1, 11)],
                    "Correct",
                    "Percent"
                ])

            student_id = str(args.student_id)

            # Prevent spreadsheet formula interpretation.

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
