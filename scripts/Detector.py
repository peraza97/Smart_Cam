import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from fractions import Fraction

RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)

predictor = dlib.shape_predictor("../models/shapes_predict.dat")


class Face:
    def __init__(self, dlib_rect, img, debugging=False):
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

    def draw_landmarks(self, img, fts):
        for pt in fts:
            pos = (pt[0,0], pt[0,1])
            cv2.circle(img, pos, 2, (255, 255, 255), -1)

    def is_smiling(self,img):
        (x,y,w,h) = self.convert_to_rect()
        ratio = self.ratio()
        if ratio > 8.5:
            cv2.rectangle(img, (x, y), (x + w, y + h), GREEN, 2)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), RED, 2)

    def draw_mouth_bbox(self, img):
        cnt = np.array(self.landmarks[48:55], dtype=np.int32)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        x,y = box[0] #bottom left
        x1,y1 = box[1] #top left
        x2,y2 = box[2] #top right
        x3,y3 = box[3] #bottom right

        self.draw_landmarks(img, self.landmarks)

        cv2.line(img,(x,y),(x1,y1),GREEN,1)
        cv2.line(img,(x1,y1),(x2,y2),GREEN,1)

        cv2.circle(img,(x,y), 3, RED, -1)
        cv2.circle(img,(x1,y1), 3, BLUE, -1)
        cv2.circle(img,(x2,y2), 3, BLUE, -1)
        cv2.circle(img,(x3,y3), 3, RED, -1)

    def grab_dims(self, fts):
        cnt = np.array(fts, dtype=np.int32)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        x,y = box[0] #bottom left
        x1,y1 = box[1] #top left
        x2,y2 = box[2] #top right
        x3,y3 = box[3] #bottom right
        h = dist.euclidean((x,y), (x1,y1))
        w = dist.euclidean((x1,y1),(x2,y2))
        if h > w:
            h,w = w,h
        return w,h

    def ratio(self):

        pt = self.landmarks[0]
        pt1 = self.landmarks[16]
        pt2 = self.landmarks[24]
        pt3 = self.landmarks[8]

        bbw = pt1[0,0] - pt[0,0]
        bbh = pt3[0,1] - pt2[0,1]

        w,h = self.grab_dims(self.landmarks[48:55])

        r = (w/bbw)/(h/bbh)
        #print("H: {}, W: {}, BBW: {}, BBH:{}".format(int(h),int(w), int(bbw),int(bbh)))
        #print("Norm ratio: {0:.2f}".format(r))
        #print()
        return r

    def eye_ratio(self):
        lw, lh = self.grab_dims(self.landmarks[36:42])
        rw, rh = self.grab_dims(self.landmarks[42:48])

        l_ratio = lh/lw
        r_ratio = rh/rw

        print("L: {}, R: {}".format(l_ratio,r_ratio))

        return l_ratio > .20 and r_ratio > .20

class Detector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def perfectPhoto(self, img, debugging=False):
        dlib_rects =  self.detector(img, 1)
        faces = [Face(rect,img,debugging) for rect in dlib_rects] 
        for face in faces:
            if not face.is_smiling(img):
                return False
        return True