import cv2
import dlib
import numpy as np
from threading import Thread
from scipy.spatial import distance as dist
from fractions import Fraction

RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)

#run video feed on a different thread
class cameraFeed():
    #init function
    def __init__(self,source=0):
        self.stream = cv2.VideoCapture(source)
        _,self.frame = self.stream.read()
        self.stopped = False
    #start function
    def start(self):
        Thread(target=self.update, args=()).start()
        return self
    #update function
    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            _,self.frame = self.stream.read()
    #read function
    def read(self):
        return self.frame
    #stop function
    def stop(self):
        self.stopped = True

#CREATE DETECTOR
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../models/shapes_predict.dat")

def grab_faces(img):
        return detector(img, 1)

def main():
    c = cameraFeed().start()

    take = True

    while True:
        frame = c.read()
        take = not take
        if not take:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dlib_rects = grab_faces(gray) #returns dliib rectangle for faces
        faces = [Face(rect,frame) for rect in dlib_rects] 
        for face in faces:
            face.is_smiling(frame)
            face.draw_mouth_bbox(frame)
            face.eye_ratio()
        cv2.imshow("Feed", frame)
        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'):
            c.stop()
            break 

        if k & 0xFF == ord('s'):
            cv2.imwrite("photos/all_smiles.png",frame)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()