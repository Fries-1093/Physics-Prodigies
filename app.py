"""
Minimal Flask server: serves the camera page, receives photos,
runs OpenCV, stores results and exports CSV.
"""

import os
import tempfile

from flask import (
    Flask,
    Response,
    jsonify,
    request,
    send_from_directory,
    render_template_string,
    redirect
)

import AnswerExport
from OpenCv import detect_answers

app = Flask(__name__)
LETTERS = set("ABCDE")


@app.route("/")
def home():
    return send_from_directory(app.root_path, "Scantron.html")


@app.route("/start")
def start():
    return send_from_directory(app.root_path, "start.html")


@app.route("/scantron")
def scantron():
    return send_from_directory(app.root_path, "Scantron.html")


@app.route("/students")
def students():
    if not AnswerExport.answer_key:
        return redirect("/start")

    return send_from_directory(app.root_path, "students.html")


@app.get("/student-count")
def student_count():
    return jsonify(
        count=len(AnswerExport.results),
        maximum=AnswerExport.MAX_STUDENTS
    )


@app.post("/scan")
def scan():

    photo = request.files.get("photo")
    mode = request.form.get("mode", "student")
    name = f"Student {len(AnswerExport.results) + 1}"

    if photo is None:
        return jsonify(error="No photo received."), 400

    if mode not in ("key", "student"):
        return jsonify(error="Invalid scanning mode."), 400

    if mode == "student":

        if not AnswerExport.answer_key:
            return jsonify(error="Scan the answer key first."), 400

        if len(AnswerExport.results) >= AnswerExport.MAX_STUDENTS:
            return jsonify(
                error="Class is full: maximum 33 students per session."
            ), 400

    # Save the uploaded photo temporarily
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)

    try:
        photo.save(path)
        answers = detect_answers(path)

    except (ValueError, OSError) as exc:
        return jsonify(error=str(exc)), 400

    finally:
        os.remove(path)

    # Process the answer key
    if mode == "key":

        if any(a not in LETTERS for a in answers):

            bad = [
                i for i, a in enumerate(answers, 1)
                if a not in LETTERS
            ]

            return jsonify(
                error=f"Answer key has unclear answers at Q{bad}. Retake the photo."
            ), 400

        AnswerExport.set_answer_key(answers)

        # Start a new marking session
        AnswerExport.results.clear()

        return jsonify(answers=answers)

   # Validate student number
    if not name.isdigit():
        return jsonify(error="Please enter a valid student number."), 400

    # Prevent duplicate student numbers
    if any(r["name"] == name for r in AnswerExport.results):
        return jsonify(error="This student number has already been scanned."), 400

    try:
        student_index = AnswerExport.add_result(name, answers)

    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    r = AnswerExport.results[student_index]

    flagged = [
        i for i, a in enumerate(answers, 1)
        if a not in LETTERS
    ]

    return jsonify(
        answers=answers,
        score=r["score"],
        total=r["total"],
        needs_review=flagged,
        student_count=len(AnswerExport.results),
        maximum=AnswerExport.MAX_STUDENTS,
        name=name
    )


@app.route("/results")
def results():

    if not AnswerExport.answer_key:
        return redirect("/start")

    return render_template_string("""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Scantroon - Results</title>

            <style>

                body{
                    text-align: center;
                    font-size: 110%;
                }

                h1{
                    margin-top: 200px;
                }

                button{
                    text-align: center;
                    background-color: rgb(19, 58, 176);
                    color: whitesmoke;
                    padding: 10px;
                    margin-bottom: 10px;
                    width: 200px;
                }

                table{
                    border-collapse: collapse;
                    margin: 20px auto;
                    width: 90%;
                    max-width: 600px;
                }

                th, td{
                    border: 1px solid black;
                    padding: 10px;
                }

                th{
                    background-color: rgb(19, 58, 176);
                    color: whitesmoke;
                }

            </style>
        </head>

        <body>

            <h1>Student Results</h1>

            <p>Total Students: {{ students | length }}</p>

            <table>

                <tr>
                    <th>Student Number</th>
                    <th>Score</th>
                    <th>Percentage</th>
                    <th>Review</th>
                </tr>

                {% for student in students %}

                <tr>

                    <td>{{ student.name }}</td>

                    <td>
                        {{ student.score }}/{{ student.total }}
                    </td>

                    <td>
                        {{ (student.score / student.total * 100) | round(1) }}%
                    </td>

                    <td>
                        {% if "REVIEW" in student.answers %}
                            Required
                        {% else %}
                            -
                        {% endif %}
                    </td>

                </tr>

                {% endfor %}

            </table>

            <br>

            <button onclick="window.location.href='/export'">
                Export CSV
            </button>

            <br>

            <button onclick="window.location.href='/students'">
                Back
            </button>

        </body>
    </html>
    """, students=AnswerExport.results)


@app.get("/export")
def export():

    return Response(
        AnswerExport.to_csv(),
        mimetype="text/csv",
        headers={
            "Content-Disposition":
            "attachment; filename=class_results.csv"
        },
    )


if __name__ == "__main__":
    app.run(debug=True)
