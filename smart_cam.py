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
    def __init__(self, dlib_rect):
        

#CREATE DETECTOR
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shapes_predict.dat")

def grab_faces(img):
        return detector(img, 1)

def convert_to_rect(dlib_rect):
            pt = dlib_rect.tl_corner()
            pt2 = dlib_rect.br_corner()
            return (pt.x,pt.y, pt2.x-pt.x,pt2.y-pt.y)

def extract_landmarks(img,faces):
    my_dict = []
    for rect in faces:
        (x,y,w,h) = convert_to_rect(rect)
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rect).parts()])
        my_dict.append( ((x,y,w,h),landmarks) ) #this stores an array
        #[ {"rect":points,"landmarsk":points}, {"rect":points,"landmarsk":points} ]
    return my_dict

def draw_features(img, fts):
    for pt in fts:
        pos = (pt[0,0], pt[0,1])
        cv2.circle(img, pos, 2, (0, 255, 255), -1)



def draw_face(img, landmarks):
    for obj in landmarks:
        (x,y,w,h) = obj[0]
        points = obj[1]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for pt in points:
            pos = (pt[0,0], pt[0,1])
            cv2.circle(img, pos, 2, (0, 255, 255), -1)

def main():
    c = cameraFeed().start()
    while True:
        frame = c.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = grab_faces(gray) #returns dliib rectangle for faces

        fts = extract_landmarks(gray, faces) #must take in dlib rectangles

        draw_face(frame,fts)

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