"""
Minimal Flask server: serves the camera page, receives photos, runs OpenCV, stores results.
Run:  pip install flask
      python app.py
Then open http://localhost:5000
Needs OpenCv.py (with detect_answers) and AnswerExport.py in the same folder.
"""
import os
import tempfile

from flask import Flask, Response, jsonify, request, send_from_directory

import AnswerExport
from OpenCv import detect_answers

app = Flask(__name__)
LETTERS = set("ABCDE")


@app.route("/")
def home():
    return send_from_directory(app.root_path, "start.html")

@app.route("/scantron")
def scantron():
    return send_from_directory(app.root_path, "Scantron.html")


@app.post("/scan")
def scan():
    photo = request.files.get("photo")
    mode = request.form.get("mode", "student")
    name = request.form.get("name", "").strip()
    if photo is None:
        return jsonify(error="No photo received."), 400

    # Save the upload to a temporary file, read it with OpenCV, then delete it
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        photo.save(path)
        answers = detect_answers(path)
    except (ValueError, OSError) as exc:
        return jsonify(error=str(exc)), 400
    finally:
        os.remove(path)

    if mode == "key":
        if any(a not in LETTERS for a in answers):
            bad = [i for i, a in enumerate(answers, 1) if a not in LETTERS]
            return jsonify(error=f"Answer key has unclear answers at Q{bad}. Retake the photo."), 400
        AnswerExport.set_answer_key(answers)
        return jsonify(answers=answers)

    if not AnswerExport.answer_key:
        return jsonify(error="Scan the answer key first."), 400
    try:
        AnswerExport.add_result(name, answers)
    except ValueError as exc:                       # class is full
        return jsonify(error=str(exc)), 400
    r = AnswerExport.results[-1]
    flagged = [i for i, a in enumerate(answers, 1) if a not in LETTERS]
    return jsonify(answers=answers, score=r["score"], total=r["total"], needs_review=flagged)


@app.get("/export")
def export():
    return Response(
        AnswerExport.to_csv(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=class_results.csv"},
    )


if __name__ == "__main__":
    app.run(debug=True)