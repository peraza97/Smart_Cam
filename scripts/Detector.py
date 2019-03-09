import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from fractions import Fraction

RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)
WHITE = (255,255,255)

predictor = dlib.shape_predictor("../models/shapes_predict.dat")

class FaceAlign:
    def __init__(self):
        pass

class Face:
    def __init__(self, dlib_rect, img):
        self.dlib_rect = dlib_rect
        self.landmarks = self.extract_landmarks(img)
    
    def extract_landmarks(self, img):
        return np.matrix([[p.x, p.y] for p in predictor(img, self.dlib_rect).parts()])

    #convert the face dlib rect to a boundingbox
    #used for drawing the rectangle around face
    def convert_to_rect(self):
        tl = self.dlib_rect.tl_corner()
        tr = self.dlib_rect.tr_corner()
        bl = self.dlib_rect.bl_corner()
        br = self.dlib_rect.br_corner()
        return (tl.x, tl.y,tr.x - tl.x , br.y-tl.y )

    #draw passed in landmarks
    def draw_landmarks(self, img, fts, color):
        for pt in fts:
            pos = (pt[0,0], pt[0,1])
            cv2.circle(img, pos, 2, color, -1)

    #from points, generate a boundingbox
    def grab_bounding_box(self,pts):
        cnt = np.array(pts, dtype=np.int32)       
        x,y,w,h = cv2.boundingRect(cnt)
        return x,y,w,h

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

    def draw_mouth_lines(self, img):
        props = self.grab_rotated_bbox(self.landmarks[48:55])
        coords = props["pts"]
        x,y = coords[0] #bottom left
        x1,y1 = coords[1] #top left
        x2,y2 = coords[2] #top right
        x3,y3 = coords[3] #bottom right
        #draw aspect ratio lines
        cv2.line(img,(x,y),(x1,y1),GREEN,1)
        cv2.line(img,(x1,y1),(x2,y2),GREEN,1)

    #helper function for is_smiling function
    def smile_ratio(self):
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
    
    #function to determine if person is smiling
    def is_smiling(self,img):
        sm_ratio = self.smile_ratio()
        return sm_ratio > 8.5

    #helper function to determine if eye_blinking
    def eye_ratio(self,eye):
        A = dist.euclidean(eye[1],eye[5])
        B = dist.euclidean(eye[2],eye[4])
        C = dist.euclidean(eye[0],eye[3])
        EAR = (A+B)/(2*C)
        return EAR

    #function to determine if someone is blinking
    def is_blinking(self, img):
        left_eye = self.eye_ratio(self.landmarks[36:42])
        right_eye = self.eye_ratio(self.landmarks[42:48])
        EAR = (left_eye + right_eye)/2.0
        return EAR < .25

    #used to determine if eyes are blinking or not
    def draw_eyes(self, img, eye_color):
        #draw the eyes
        lcnt = np.array(self.landmarks[36:42], dtype=np.int32)
        rcnt = np.array(self.landmarks[42:48], dtype=np.int32)
        cv2.drawContours(img,[lcnt],0,eye_color,1)
        cv2.drawContours(img,[rcnt],0,eye_color,1)

    #used to detemine if face is smiling or not
    def draw_face_bbox(self, img, box_color):
        #draw box around eyes
        x,y,w,h = self.convert_to_rect()
        cv2.rectangle(img, (x, y), (x + w, y + h), box_color, 2)
        

class Detector:
    def __init__(self, debugging):
        self.detector = dlib.get_frontal_face_detector()
        self.debugging = debugging
        if self.debugging is None:
            self.debugging = "None"
        print(self.debugging)

    def perfectPhoto(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dlib_rects =  self.detector(gray, 1) #grab all rectangles
        faces = [Face(rect,img) for rect in dlib_rects] 
        for face in faces:
            smiling = face.is_smiling(img)
            blinking = face.is_blinking(img)

            #THIS IS FOR DEBUGGING
            eye_color = RED if blinking else GREEN
            box_color = GREEN if smiling else RED
            if self.debugging == "eyes":
                face.draw_eyes(img, eye_color)
            elif self.debugging == "face":
                face.draw_face_bbox(img, box_color)
            elif self.debugging == "both":
                face.draw_eyes(img, eye_color)
                face.draw_face_bbox(img, box_color)
            
        return smiling and not blinking