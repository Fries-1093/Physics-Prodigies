
"""
10-question, two-column Scantron reader.

Features:
- Automatic paper-corner detection
- Perspective correction
- Automatic rotation detection
- More flexible bubble detection
- Answer classification
- REVIEW detection
- Debug image output
- Manual answer corrections
"""

import argparse
import csv
import itertools
from pathlib import Path

import cv2
import numpy as np


LETTERS = "ABCDE"
N_QUESTIONS = 10
OPTIONS = 5

WORK_WIDTH = 850

PAGE_WIDTH = 850
PAGE_HEIGHT = 1100

MAX_SHADED_STD = 27.0
MIN_DARKER_THAN_OTHERS = 12.0

REVIEW_STD = 42.0
REVIEW_DARKNESS = 20.0

SECOND_MARK_STD = 38.0
SECOND_MARK_DARKNESS = 18.0


# --------------------------------------------------
# PAPER DETECTION
# --------------------------------------------------

def order_points(points):
    """Order corners: top-left, top-right, bottom-right, bottom-left."""

    points = np.array(points, dtype="float32")

    ordered = np.zeros((4, 2), dtype="float32")

    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).flatten()

    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]

    return ordered


def find_paper_corners(img):
    """Find the four corners of the paper."""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, threshold = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((5, 5), np.uint8)

    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = img.shape[0] * img.shape[1]

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < image_area * 0.15:
            continue

        perimeter = cv2.arcLength(contour, True)

        for epsilon in (0.01, 0.015, 0.02, 0.03, 0.04):

            approx = cv2.approxPolyDP(
                contour,
                epsilon * perimeter,
                True
            )

            if len(approx) == 4 and cv2.isContourConvex(approx):
                return order_points(approx.reshape(4, 2))

    # Second attempt using edge detection.
    edges = cv2.Canny(blur, 30, 100)

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < image_area * 0.15:
            continue

        perimeter = cv2.arcLength(contour, True)

        for epsilon in (0.015, 0.02, 0.03, 0.04, 0.05):

            approx = cv2.approxPolyDP(
                contour,
                epsilon * perimeter,
                True
            )

            if len(approx) == 4 and cv2.isContourConvex(approx):
                return order_points(approx.reshape(4, 2))

    raise ValueError(
        "Could not detect the four paper corners. "
        "Make sure the entire paper is visible."
    )


# --------------------------------------------------
# PERSPECTIVE CORRECTION
# --------------------------------------------------

def correct_perspective(img):
    """Straighten the paper while preserving its proportions."""

    corners = find_paper_corners(img)

    top_left, top_right, bottom_right, bottom_left = corners

    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)

    height_left = np.linalg.norm(bottom_left - top_left)
    height_right = np.linalg.norm(bottom_right - top_right)

    paper_width = max(width_top, width_bottom)
    paper_height = max(height_left, height_right)

    if paper_width >= paper_height:
        output_width = PAGE_HEIGHT
        output_height = PAGE_WIDTH
    else:
        output_width = PAGE_WIDTH
        output_height = PAGE_HEIGHT

    destination = np.array([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(
        corners,
        destination
    )

    corrected = cv2.warpPerspective(
        img,
        matrix,
        (output_width, output_height)
    )

    return corrected


# --------------------------------------------------
# RESIZE IMAGE
# --------------------------------------------------

def resize_page(img):

    height = round(
        img.shape[0] * WORK_WIDTH / img.shape[1]
    )

    return cv2.resize(
        img,
        (WORK_WIDTH, height),
        interpolation=cv2.INTER_AREA
    )


# --------------------------------------------------
# ROTATION DETECTION
# --------------------------------------------------

def find_correct_orientation(img):
    """
    Test four rotations and select the orientation
    in which the expected question layout is detected.
    """

    rotations = [
        ("0 degrees", img),
        (
            "90 degrees clockwise",
            cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        ),
        (
            "180 degrees",
            cv2.rotate(img, cv2.ROTATE_180)
        ),
        (
            "90 degrees counterclockwise",
            cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        )
    ]

    errors = []

    for label, rotated in rotations:

        candidate = resize_page(rotated)

        gray = cv2.cvtColor(
            candidate,
            cv2.COLOR_BGR2GRAY
        )

        try:

            rows = find_grid(gray)

            print("Detected page orientation:", label)

            return candidate, gray, rows

        except ValueError as exc:

            errors.append(f"{label}: {exc}")

    raise ValueError(
        "Could not find the 50 bubbles in any orientation.\n"
        + "\n".join(errors)
    )


# --------------------------------------------------
# BUBBLE DETECTION
# --------------------------------------------------

def find_grid(gray):
    """
    Find the ten question rows.

    Each question row contains:
    - Five bubbles in the left column
    - Five bubbles in the right column

    The output order is Q1-Q5 on the left,
    followed by Q6-Q10 on the right.
    """

    # Improve contrast in darker photographs.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    blur = cv2.GaussianBlur(
        enhanced,
        (5, 5),
        1
    )

    # Detect circles.
    found = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=18,
        param1=100,
        param2=17,
        minRadius=5,
        maxRadius=17
    )

    if found is None:
        raise ValueError(
            "No bubbles detected. Try a brighter, clearer photo."
        )

    circles = found[0].tolist()

    if len(circles) < 50:
        raise ValueError(
            f"Only detected {len(circles)} circles. "
            "Try a clearer photo with all bubbles visible."
        )

    # --------------------------------------------------
    # GROUP CIRCLES INTO HORIZONTAL ROWS
    # --------------------------------------------------

    circles.sort(key=lambda c: c[1])

    row_groups = []

    for circle in circles:

        x, y, radius = circle

        placed = False

        for group in row_groups:

            average_y = np.median([
                c[1] for c in group
            ])

            if abs(y - average_y) <= 16:
                group.append(circle)
                placed = True
                break

        if not placed:
            row_groups.append([circle])

    # Ignore small groups of false detections.
    possible_rows = [
        group for group in row_groups
        if len(group) >= 8
    ]

    if len(possible_rows) < 5:
        raise ValueError(
            f"Detected only {len(possible_rows)} question rows "
            "instead of 5. Try a brighter photo with the entire "
            "answer area visible."
        )

    # Keep the five strongest rows.
    possible_rows.sort(
        key=lambda group: len(group),
        reverse=True
    )

    selected_rows = possible_rows[:5]

    # Restore top-to-bottom order.
    selected_rows.sort(
        key=lambda group: np.median([
            c[1] for c in group
        ])
    )

    # --------------------------------------------------
    # SELECT FIVE BUBBLES IN EACH COLUMN
    # --------------------------------------------------

    def select_five(group):

        group = sorted(
            group,
            key=lambda c: c[0]
        )

        if len(group) < 5:
            raise ValueError(
                "A question row has fewer than five detected bubbles."
            )

        if len(group) > 5:

            best_group = None
            best_score = float("inf")

            for combination in itertools.combinations(group, 5):

                xs = [
                    c[0] for c in combination
                ]

                gaps = np.diff(xs)

                if np.any(gaps < 15):
                    continue

                score = float(np.std(gaps))

                if score < best_score:
                    best_score = score
                    best_group = combination

            if best_group is None:
                raise ValueError(
                    "Could not identify five evenly spaced bubbles."
                )

            group = list(best_group)

        return sorted(
            group,
            key=lambda c: c[0]
        )

    # --------------------------------------------------
    # SEPARATE LEFT AND RIGHT COLUMNS
    # --------------------------------------------------

    left_column = []
    right_column = []

    for group in selected_rows:

        group = sorted(
            group,
            key=lambda c: c[0]
        )

        gaps = [
            group[i + 1][0] - group[i][0]
            for i in range(len(group) - 1)
        ]

        split_index = int(np.argmax(gaps))

        left_candidates = group[:split_index + 1]
        right_candidates = group[split_index + 1:]

        if (
            len(left_candidates) < 5
            or len(right_candidates) < 5
        ):
            raise ValueError(
                "Could not separate the two answer columns."
            )

        left = select_five(left_candidates)
        right = select_five(right_candidates)

        left_y = np.median([
            c[1] for c in left
        ])

        right_y = np.median([
            c[1] for c in right
        ])

        if abs(left_y - right_y) > 18:
            raise ValueError(
                "The two answer columns are not aligned."
            )

        left_column.append(left)
        right_column.append(right)

    final_rows = left_column + right_column

    if len(final_rows) != 10:
        raise ValueError(
            "Could not identify all ten question rows."
        )

    return final_rows


# --------------------------------------------------
# BUBBLE MEASUREMENT
# --------------------------------------------------

def measure(gray, circle):
    """Measure the central shading of a bubble."""

    x, y, _ = (
        int(round(value))
        for value in circle
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

    pixels = patch[
        xx * xx + yy * yy < 40
    ]

    return (
        float(np.std(pixels)),
        float(np.median(pixels))
    )


# --------------------------------------------------
# ANSWER CLASSIFICATION
# --------------------------------------------------

def classify_question(stats):

    stds = [s for s, _ in stats]
    medians = [m for _, m in stats]

    baseline = float(np.median(medians))

    strong = []

    for i in range(OPTIONS):

        others = [
            medians[j]
            for j in range(OPTIONS)
            if j != i
        ]

        if (
            stds[i] < MAX_SHADED_STD
            and np.median(others) - medians[i]
            >= MIN_DARKER_THAN_OTHERS
        ):
            strong.append(i)

    possible = [
        i for i in range(OPTIONS)
        if i not in strong
        and stds[i] < SECOND_MARK_STD
        and baseline - medians[i] >= SECOND_MARK_DARKNESS
    ]

    if len(strong) >= 2:

        answer = "MULTIPLE"

    elif len(strong) == 1:

        answer = "REVIEW" if possible else LETTERS[strong[0]]

    elif len(possible) >= 2:

        answer = "REVIEW"

    elif len(possible) == 1:

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

        answer = (
            "REVIEW"
            if delta >= REVIEW_DARKNESS
            and stds[darkest] < REVIEW_STD
            else "BLANK"
        )

    return answer, strong, possible


# --------------------------------------------------
# MAIN DETECTION FUNCTION
# --------------------------------------------------

def detect_answers(path, debug_path="debug_output_photo.png"):

    img = cv2.imread(str(path))

    if img is None:
        raise ValueError(
            f"Could not open image: {path}"
        )

    # STEP 1: Detect paper and straighten it.
    corrected = correct_perspective(img)

    # STEP 2: Find the correct page orientation.
    img, gray, rows = find_correct_orientation(corrected)

    # STEP 3: Prepare debug image.
    annotated = img.copy()

    answers = []

    print("Question | Answer   | Std deviations      | Medians")
    print("-" * 76)

    # STEP 4: Read each question.
    for q, row in enumerate(rows, start=1):

        stats = [
            measure(gray, circle)
            for circle in row
        ]

        answer, strong, possible = classify_question(stats)

        answers.append(answer)

        for i, (x, y, _) in enumerate(row):

            color = (
                (0, 190, 0) if i in strong
                else (0, 165, 255) if i in possible
                else (255, 0, 0)
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
            (max(0, int(x) - 55), max(12, int(y) - 16)),
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

    # STEP 5: Save debug image.
    if not cv2.imwrite(str(debug_path), annotated):
        raise ValueError(
            f"Could not save debug image: {debug_path}"
        )

    print("\nANSWERS:", answers)
    print("Debug image:", debug_path)

    return answers


# --------------------------------------------------
# MANUAL ANSWER CORRECTIONS
# --------------------------------------------------

def apply_corrections(answers, entries, label):

    updated = answers.copy()
    used = set()

    for entry in entries:

        try:

            question_text, answer = entry.split("=", 1)
            question = int(question_text)

        except ValueError as exc:

            raise ValueError(
                f"Invalid {label} correction '{entry}'; use 3=B."
            ) from exc

        answer = answer.strip().upper()

        if question not in range(1, N_QUESTIONS + 1) or question in used:
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


# --------------------------------------------------
# COMMAND-LINE TESTING
# --------------------------------------------------

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
        metavar="Q=ANSWER"
    )

    parser.add_argument(
        "--key-set",
        action="append",
        default=[],
        metavar="Q=ANSWER"
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
                print("Corrected answers:", student)

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
                for i, answer in enumerate(student, 1)
                if answer == "REVIEW"
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
