import cv2
import urllib
import pdb
import numpy as np

import picamera as PiCamera

class VideoCamera(object):

    self.camera = PiCamera()
    self.rawCapture = None

    def __init__(self):
        self.camera.resolution = (640, 480)
        self.camera.framerate = 32
        self.rawCapture = PiRGBArray(self.camera, size=(640, 480))
        time.sleep(0.1)

    def __del__(self):
        self.rawCapture.truncate(0)

    def get_frame(self):
        frame = self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
