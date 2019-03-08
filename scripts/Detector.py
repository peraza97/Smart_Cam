import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from fractions import Fraction

RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)

predictor = dlib.shape_predictor("../models/shapes_predict.dat")

class FaceAlign:
    def __init__(self):
        pass

class Face:
    def __init__(self, dlib_rect, img, debugging=False):
        self.debugging = debugging
        self.dlib_rect = dlib_rect
        self.landmarks = self.extract_landmarks(img)
    
    def extract_landmarks(self, img):
        x,y,w,h = self.convert_to_rect()
        tmp = img[y:y+h,x:x+w]
        cv2.imshow("cropped", tmp)
        cv2.waitKey()
        return np.matrix([[p.x, p.y] for p in predictor(tmp).parts()])

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

    def grab_rotated_bbox(self,pts):
        cnt = np.array(pts, dtype=np.int32)
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

        props = {"pts": [(x,y), (x1,y1), (x2,y2), (x3,y3)], "w":w, "h":h}
        #return properties of the bounding box
        return props 

    def draw_mouth_bbox(self, img):
        props = self.grab_rotated_bbox(self.landmarks[48:55])
        coords = props["pts"]
        x,y = coords[0] #bottom left
        x1,y1 = coords[1] #top left
        x2,y2 = coords[2] #top right
        x3,y3 = coords[3] #bottom right

        #draw aspect ratio lines
        cv2.line(img,(x,y),(x1,y1),GREEN,1)
        cv2.line(img,(x1,y1),(x2,y2),GREEN,1)

    def ratio(self):
        pt = self.landmarks[0]
        pt1 = self.landmarks[16]
        pt2 = self.landmarks[24]
        pt3 = self.landmarks[8]

        bbw = pt1[0,0] - pt[0,0]
        bbh = pt3[0,1] - pt2[0,1]

        props = self.grab_rotated_bbox(self.landmarks[48:55])
        w = props["w"]
        h = props["h"]
        r = (w/bbw)/(h/bbh)
        return r

    def is_smiling(self,img):
        ratio = self.ratio()
        if not self.debugging:
            return ratio > 8.5

        self.draw_landmarks(img,self.landmarks)
        self.draw_mouth_bbox(img)
        (x,y,w,h) = self.convert_to_rect()
        if ratio > 8.5:
            cv2.rectangle(img, (x, y), (x + w, y + h), GREEN, 2)
            return True
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), RED, 2)
            return False

class Detector:
    def __init__(self,debugging=False):
        self.detector = dlib.get_frontal_face_detector()
        sp = dlib.shape_predictor("../models/shapes_5.dat")
        self.debugging = debugging

    def perfectPhoto(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dlib_rects =  self.detector(gray, 1) #grab all rectangles
        faces = [Face(rect,img,self.debugging) for rect in dlib_rects] 
        for face in faces:
            if not face.is_smiling(img):
                return False
        return True