import cv2
import dlib
import numpy as np
from threading import Thread


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

class Face:
    def __init__(self, dlib_rect, img):
        self.dlib_rect = dlib_rect
        self.landmarks = self.extract_landmarks(img)

    def extract_landmarks(self, img):
        return np.matrix([[p.x, p.y] for p in predictor(img, self.dlib_rect).parts()])

    def convert_to_rect(self):
        pt = self.dlib_rect.tl_corner()
        pt2 = self.dlib_rect.br_corner()
        return (pt.x,pt.y, pt2.x-pt.x,pt2.y-pt.y)

    def detect_smile():
        fts = self.landmarks[48:68]
        ##do our shit

    def draw_face(self,img):
        (x,y,w,h) = self.convert_to_rect()
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def draw_landmarks(self, img):
        for pt in self.landmarks:
            pos = (pt[0,0], pt[0,1])
            cv2.circle(img, pos, 2, (0, 255, 255), -1)

    def draw_smile(self,img):
        fts = self.landmarks[48:68]
        for pt in fts:
            pos = (pt[0,0], pt[0,1])
            cv2.circle(img, pos, 2, (0, 255, 255), -1)



#CREATE DETECTOR
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shapes_predict.dat")

def grab_faces(img):
        return detector(img, 1)

def main():
    c = cameraFeed().start()
    while True:
        frame = c.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        dlib_rects = grab_faces(gray) #returns dliib rectangle for faces

        faces = [Face(rect,frame) for rect in dlib_rects] 
        for face in faces:
            face.draw_smile(frame)

        cv2.imshow("Feed", frame)
        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'):
            c.stop()
            break 
        elif k & 0xFF == ord('s'):
            cv2.imwrite( "./photos/detectedLandmarks.jpg", frame);

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()