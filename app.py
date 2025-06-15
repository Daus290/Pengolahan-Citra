# main program

import os
import cv2
import uuid

from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from tugas.edge_detection import apply_edge_detection
from tugas.morfologi import apply_morphology
from tugas.morfologi import apply_morphology_lanjutan

app = Flask(__name__)
app.secret_key = 'daus_rahasia'


@app.route('/')
def index():
    return render_template('index.html', request=request)


# tugas 1
@app.route("/tugas1", methods=["GET", "POST"])
def tugas1():
    threshold = 127
    filename = None

    if request.method == "POST":
        threshold = int(request.form.get("threshold", 127))
        
        if "image" in request.files and request.files["image"].filename != "":
            image_file = request.files["image"]
            filename = secure_filename(image_file.filename)
            filepath = os.path.join("static", filename)
            image_file.save(filepath)
        elif "filename" in request.form:
            filename = request.form["filename"]

        if filename:
            # Proses thresholding
            img = cv2.imread(os.path.join("static", filename), cv2.IMREAD_GRAYSCALE)
            _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
            cv2.imwrite("static/threshold_result.jpg", binary)

    return render_template("tugas1.html", filename=filename, threshold=threshold)


#tugas 2
@app.route('/tugas2', methods=['GET', 'POST'])
def tugas2():
    operator = request.args.get('operator', 'sobel')

    if request.method == 'GET' and 'operator' not in request.args:
        session.pop('filename', None)

    filename = session.get('filename')
    threshold1 = 100
    threshold2 = 200

    if request.method == 'POST':
        file = request.files.get('image')
        form_filename = request.form.get('filename')
        threshold1 = int(request.form.get('threshold1', 100))
        threshold2 = int(request.form.get('threshold2', 200))

        if file and file.filename != '':
            ext = os.path.splitext(file.filename)[1]
            unique_name = f"{uuid.uuid4().hex}{ext}"
            path = os.path.join('static', unique_name)
            file.save(path)

            session['filename'] = unique_name
            filename = unique_name
        elif form_filename:
            filename = form_filename
            path = os.path.join('static', filename)
        else:
            path = None

        if path:
            results = apply_edge_detection(path, threshold1, threshold2)
            for key, img in results.items():
                cv2.imwrite(f'static/{key}.jpg', img)

        return render_template('tugas2.html',
                               filename=filename,
                               operator=operator,
                               threshold1=threshold1,
                               threshold2=threshold2,
                               request=request)

    return render_template('tugas2.html',
                           filename=filename,
                           operator=operator,
                           threshold1=threshold1,
                           threshold2=threshold2,
                           request=request)

#tugas 3
@app.route('/tugas3', methods=['GET', 'POST'])
def tugas3():
    operation = request.form.get('operation', 'erode')
    shape = request.form.get('shape', 'rectangle')
    filename = session.get('filename')

    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename != '':
            ext = os.path.splitext(file.filename)[1]
            unique_name = f"{uuid.uuid4().hex}{ext}"
            path = os.path.join('static', unique_name)
            file.save(path)
            session['filename'] = unique_name
            filename = unique_name
        elif filename:
            path = os.path.join('static', filename)
        else:
            path = None

        if path:
            result = apply_morphology(path, operation, shape)
            cv2.imwrite('static/morph_result.jpg', result)

    return render_template('tugas3.html', filename=filename, operation=operation, shape=shape)

# tugas 4
@app.route('/tugas4', methods=['GET', 'POST'])
def tugas4():
    operation = request.form.get('operation', 'boundary')
    outline_only = request.form.get('outline') == 'on'
    filename = session.get('filename')

    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename != '':
            ext = os.path.splitext(file.filename)[1]
            unique_name = f"{uuid.uuid4().hex}{ext}"
            path = os.path.join('static', unique_name)
            file.save(path)
            session['filename'] = unique_name
            filename = unique_name
        elif filename:
            path = os.path.join('static', filename)
        else:
            path = None

        if path:
            result = apply_morphology_lanjutan(path, operation, outline_only)
            cv2.imwrite('static/morph_lanjutan_result.jpg', result)

    return render_template('tugas4.html', filename=filename, operation=operation, outline_only=outline_only)

if __name__ == '__main__':
    app.run(debug=True, port=8888)