import os
import tempfile
from threading import Lock

import cv2

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
scan_lock = Lock()


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

    with scan_lock:

        return jsonify(
            count=len(AnswerExport.results),
            maximum=AnswerExport.MAX_STUDENTS,
            student_count=len(AnswerExport.results),
            max_students=AnswerExport.MAX_STUDENTS
        )


@app.post("/scan")
def scan():

    photo = request.files.get("photo")
    mode = request.form.get("mode", "student")

    if photo is None:
        return jsonify(error="No photo received."), 400

    if mode not in ("key", "student"):
        return jsonify(error="Invalid scanning mode."), 400

    with scan_lock:

        if mode == "student":

            if not AnswerExport.answer_key:
                return jsonify(
                    error="Scan the answer key first."
                ), 400

            if len(AnswerExport.results) >= AnswerExport.MAX_STUDENTS:

                return jsonify(
                    error="Maximum number of students reached."
                ), 400

        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)

        try:

            photo.save(path)
            answers = detect_answers(path)

        except (ValueError, OSError, cv2.error) as exc:

            return jsonify(error=str(exc)), 400

        finally:

            if os.path.exists(path):
                os.remove(path)

        # Save answer key and reset previous results

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

            AnswerExport.results.clear()

            return jsonify(
                answers=answers,
                student_count=0,
                count=0,
                maximum=AnswerExport.MAX_STUDENTS,
                max_students=AnswerExport.MAX_STUDENTS
            )

        # Automatically assign an internal student name

        name = f"Student {len(AnswerExport.results) + 1}"

        try:

            student_index = AnswerExport.add_result(
                name,
                answers
            )

        except ValueError as exc:

            return jsonify(error=str(exc)), 400

        result = AnswerExport.results[student_index]

        flagged = [
            i for i, a in enumerate(answers, 1)
            if a not in LETTERS
        ]

        return jsonify(
            answers=answers,
            score=result["score"],
            total=result["total"],
            needs_review=flagged,
            student_count=len(AnswerExport.results),
            count=len(AnswerExport.results),
            maximum=AnswerExport.MAX_STUDENTS,
            max_students=AnswerExport.MAX_STUDENTS,
            name=name
        )


@app.get("/results")
def results():

    with scan_lock:

        if not AnswerExport.answer_key:
            return redirect("/start")

        return render_template_string("""

<!DOCTYPE html>
<html>
<head>

    <title>MCQ Scanner - Results</title>

    <meta name="viewport"
          content="width=device-width, initial-scale=1.0">

    <style>

        body {
            text-align: center;
            font-family: Arial, sans-serif;
            font-size: 150%;
        }

        h1 {
            margin-top: 8%;
            font-size: 70px;
            margin-bottom: 30px;
        }

        button {
            background-color: rgb(19, 58, 176);
            color: whitesmoke;
            padding: 20px;
            margin-bottom: 25px;
            width: 300px;
            max-width: 90%;
            font-size: 30px;
            font-weight: bold;
            border: none;
            border-radius: 10px;
            cursor: pointer;
        }

        table {
            border-collapse: collapse;
            margin: 30px auto;
            width: 90%;
            max-width: 1000px;
            font-size: 24px;
        }

        th, td {
            border: 1px solid black;
            padding: 16px;
        }

        th {
            background-color: rgb(19, 58, 176);
            color: whitesmoke;
        }

        #totalStudents {
            font-size: 30px;
            font-weight: bold;
        }

        @media (max-width: 600px) {

            h1 {
                font-size: 42px;
            }

            button {
                font-size: 24px;
            }

            table {
                font-size: 16px;
            }

            th, td {
                padding: 10px;
            }

            #totalStudents {
                font-size: 24px;
            }

        }

    </style>

</head>

<body>

    <h1>Student Results</h1>

    <p id="totalStudents">
        Total Students: {{ students | length }}
    </p>

    <table>

        <tr>
            <th>Score</th>
            <th>Percentage</th>
            <th>Review</th>
        </tr>

        {% for student in students %}

        <tr>

            <td>
                {{ student.score }}/{{ student.total }}
            </td>

            <td>
                {{ (student.score / student.total * 100) | round(1) }}%
            </td>

            <td>
                {% if "REVIEW" in student.answers %}
                    Required (provisional)
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

    with scan_lock:

        return Response(
            AnswerExport.to_csv(),
            mimetype="text/csv",
            headers={
                "Content-Disposition":
                "attachment; filename=class_results.csv"
            }
        )


if __name__ == "__main__":

    app.run(debug=True)
