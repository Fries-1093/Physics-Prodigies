"""
Bubble detection for a 10-question, 5-option (A-E) Scantron sheet.
Usage: python detect_bubbles.py photo.jpg
Requires: pip install opencv-python numpy
Assumes one column of 10 rows, 5 bubbles per row.
"""
import sys
import cv2
import numpy as np

N_Q, N_OPT = 10, 5
LETTERS = "ABCDE"

# --- Tunable parameters -------------------------------------------------
WORK_WIDTH = 1000      # image is resized to this width before processing
MIN_SIZE = 0.015       # min bubble width as fraction of image width
MAX_SIZE = 0.060       # max bubble width as fraction of image width
ASPECT_RANGE = (0.75, 1.30)   # width/height range for a "round" bubble
FILL_THRESHOLD = 0.45  # fill ratio above this = bubble is marked
# ------------------------------------------------------------------------


def order_points(pts):
    pts = pts.reshape(4, 2).astype("float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32")


def warp_paper(img):
    """Try to find the paper outline and flatten it. Falls back to original."""
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, None, iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.3 * img.shape[0] * img.shape[1]:
            rect = order_points(approx)
            (tl, tr, br, bl) = rect
            w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
            h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
            dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            return cv2.warpPerspective(img, M, (w, h))
    return img


def find_bubbles(thresh, width):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubbles = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if not (MIN_SIZE * width <= w <= MAX_SIZE * width):
            continue
        if not (ASPECT_RANGE[0] <= w / float(h) <= ASPECT_RANGE[1]):
            continue
        bubbles.append((x, y, w, h))
    return bubbles


def group_into_grid(bubbles):
    bubbles = sorted(bubbles, key=lambda b: b[1] + b[3] / 2)   # top to bottom
    rows = []
    for i in range(0, len(bubbles), N_OPT):
        row = sorted(bubbles[i:i + N_OPT], key=lambda b: b[0])  # left to right
        rows.append(row)
    return rows


def fill_ratio(thresh, bubble):
    x, y, w, h = bubble
    cx, cy = x + w // 2, y + h // 2
    r = int(min(w, h) * 0.30)           # sample only the center, skip the ring
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.circle(mask, (cx, cy), r, 255, -1)
    filled = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
    return filled / float(cv2.countNonZero(mask))


def main(path):
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Could not read image: {path}")

    img = warp_paper(img)
    scale = WORK_WIDTH / img.shape[1]
    img = cv2.resize(img, None, fx=scale, fy=scale)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    bubbles = find_bubbles(thresh, img.shape[1])
    print(f"Bubbles found: {len(bubbles)} (expected {N_Q * N_OPT})")

    debug = img.copy()
    for (x, y, w, h) in bubbles:
        cv2.rectangle(debug, (x, y), (x + w, y + h), (255, 0, 0), 1)   # blue = candidate

    if len(bubbles) != N_Q * N_OPT:
        print("WARNING: wrong bubble count. Adjust MIN_SIZE / MAX_SIZE and check debug_output.png.")
        cv2.imwrite("debug_output.png", debug)
        cv2.imwrite("debug_threshold.png", thresh)
        return

    rows = group_into_grid(bubbles)
    answers = []
    for q, row in enumerate(rows, start=1):
        ratios = [fill_ratio(thresh, b) for b in row]
        marked = [i for i, r in enumerate(ratios) if r >= FILL_THRESHOLD]
        if len(marked) == 0:
            result = "BLANK"
        elif len(marked) > 1:
            result = "MULTIPLE"
        else:
            result = LETTERS[marked[0]]
            x, y, w, h = row[marked[0]]
            cv2.circle(debug, (x + w // 2, y + h // 2), w // 2 + 3, (0, 200, 0), 3)  # green = detected
        answers.append(result)
        print(f"Q{q:>2}: {result:<8} ratios = {[round(r, 2) for r in ratios]}")

    print("\nAnswers:", answers)
    cv2.imwrite("debug_output.png", debug)
    cv2.imwrite("debug_threshold.png", thresh)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python detect_bubbles.py photo.jpg")
    main(sys.argv[1])
