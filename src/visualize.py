import os
from typing import List

import cv2
import cv2.cv2 as cv2
import dlib
import numpy as np
import requests
from imutils import face_utils


class Detector:
    def __init__(self):
        self._detector = dlib.get_frontal_face_detector()

    def detect(self, image: np.ndarray) -> dlib.rectangle:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self._detector(image, 0)

    def plot(self, image: np.ndarray, bboxes: dlib.rectangle) -> np.ndarray:
        pass


class KeyPointPredictor:
    def __init__(self):
        cache_dir = os.path.join(os.path.expanduser('~'), '.lipread')
        os.makedirs(cache_dir, exist_ok=True)
        data = os.path.join(cache_dir, 'shape_predictor_68_face_landmarks.dat')
        if not os.path.exists(data):
            print(f'Download data for {self.__class__.__name__}')
            url = 'https://github.com/AKSHAYUBHAT/TensorFace/raw/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat'
            r = requests.get(url, allow_redirects=True)
            open(data, 'wb').write(r.content)

        self._prediction = dlib.shape_predictor(data)

    def predict(self, image: np.ndarray, bbox: dlib.rectangle):
        return self._prediction(image, bbox)


detector = Detector()
predictor = KeyPointPredictor()


def debug_window(show_bbox: bool = True, show_keypoints: bool = True, show_fps: bool = True):
    detector = Detector()
    predictor = KeyPointPredictor()

    cap = cv2.VideoCapture(0)
    while True:
        _, image = cap.read()
        rects = detector.detect(image)
        for (i, rect) in enumerate(rects):
            shape = predictor.predict(image, rect)
            shape = face_utils.shape_to_np(shape)
            for (x, y) in shape:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        cv2.imshow("Output", image)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    debug_window()
