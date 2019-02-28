import cv2
import dlib
import numpy as np
from threading import Thread
from scipy.spatial import distance as dist
import argparse as argp
from fractions import Fraction

RED = (0,0,255)
GREEN = (0,255,0)

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
predictor = dlib.shape_predictor("models/shapes_predict.dat")

def grab_faces(img):
        return detector(img, 1)

def draw_rect(img, x,y,w,h,color):
    cv2.rectangle(img, (x, y), (x + w, y + h),color, 2)


class Face:
    def __init__(self, dlib_rect, img):
        self.dlib_rect = dlib_rect
        self.landmarks = self.extract_landmarks(img)
        self.bbox = self.convert_to_rect()

    def extract_landmarks(self, img):
        return np.matrix([[p.x, p.y] for p in predictor(img, self.dlib_rect).parts()])

    def convert_to_rect(self):
        tl = self.dlib_rect.tl_corner()
        tr = self.dlib_rect.tr_corner()
        bl = self.dlib_rect.bl_corner()
        br = self.dlib_rect.br_corner()
        return (tl.x, tl.y,tr.x - tl.x , br.y-tl.y )

    def draw_whole_landmarks(self,img):
        fts = self.landmarks 
        #smile is self.landmarks[48:68] 
        for pt in fts:
            pos = (pt[0,0], pt[0,1])
            cv2.circle(img, pos, 2, (0, 255, 255), -1)

    def draw_landmarks(self, img, fts):
        for pt in fts:
            pos = (pt[0,0], pt[0,1])
            cv2.circle(img, pos, 2, (0, 255, 255), -1)


    def is_smiling(self,img):
        (x,y,w,h) = self.convert_to_rect()
        ratio = self.ratio()
        if ratio > 10:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    def draw_upperlip(self, img):
        points = self.landmarks[48:55]
        cnt = np.array(points, dtype=np.int32)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img,[box],0,(0,0,255),2)

    def ratio(self):
        points = self.landmarks[48:55]
        cnt = np.array(points, dtype=np.int32)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        x,y = box[0] #bottom left
        x1,y1 = box[1] #top left
        x2,y2 = box[2] #top right
        x3,y3 = box[3] #bottom right

        h = dist.euclidean((x,y), (x1,y1))
        w = dist.euclidean((x1,y1),(x2,y2))
        bbx,bby,bbw,bbh = self.convert_to_rect()
        r = (w/bbw)/(h/bbh)
        print("H: {}, W: {}".format(h,w))
        print("BBW: {}, BBH: {}".format(bbw,bbh))
        print("Normalize, W/H: {}".format(r ))
        return r

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