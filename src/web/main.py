import sys
sys.path.append('/Users/alexon/Projects/bachelor-thesis/')

import cv2
from flask import Flask, Response, render_template

from camera import VideoCamera
from src.visualize import Detector, KeyPointPredictor, plot_points



app = Flask(__name__)

detector = Detector()
predictor = KeyPointPredictor()


@app.route('/')
def index():
    return render_template('index.html')


def gen(camera):
    while True:
        image = camera.get_frame()
        image = plot_points(detector, predictor, image)
        _, jpeg = cv2.imencode('.jpg', image)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
