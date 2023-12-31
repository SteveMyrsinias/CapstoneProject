from flask import Flask, render_template, url_for, Response, request, redirect, session
from flask_socketio import SocketIO, emit
from markupsafe import escape
import cv2
import base64
from AVRA.vehicle_detection import traffic_analyser
from ALPR.tracking import licencePlateDetection
import numpy as np
import json
import os

app = Flask(__name__)
socketio = SocketIO(app)
upload_folder = "Uploaded Videos"
app.secret_key = "RandomString123"
app.config["SESSION_PERMANENT"] = False


@app.route('/render')
def home():
    summary = {}
    with open("log.json", "r+") as f:
        summary = json.load(f)
    return render_template("index.html", summary=summary)


@app.route('/feed')
def feed():
    return Response(traffic_analyser(session['filename']), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/feed2')
def feed2():
    return Response(licencePlateDetection(model_name='ALPR/best.pt', filename=session['filename']),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/', methods=["POST", "GET"])
def uploader():
    if request.method == "POST":
        f = request.files['file']
        if f:
            filename = os.path.join(upload_folder, f.filename)
            session["filename"] = filename
            f.save(filename)
        else:
            session["filename"] = os.path.join(upload_folder, "bridge.mp4")
        return redirect(url_for('home'))

    with open("log.json", "r+") as f:
        summary = json.load(f)

    return render_template("upload.html", summary=summary)


if __name__ == '__main__':
    socketio.run(app, port=4000, debug=True, allow_unsafe_werkzeug=True)
