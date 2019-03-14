import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from fractions import Fraction
import glob
import os

RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)
WHITE = (255,255,255)
ORANGE = (0,162,255)

predictor = dlib.shape_predictor("../models/shapes_predict.dat")

class Face:
    def __init__(self, dlib_rect, img):
        self.dlib_rect = dlib_rect
        self.landmarks = np.matrix([[p.x, p.y] for p in predictor(img, self.dlib_rect).parts()])
        pt = self.landmarks[0]
        pt1 = self.landmarks[16]
        pt2 = self.landmarks[24]
        pt3 = self.landmarks[8]
        self.w = pt1[0,0] - pt[0,0]
        self.h = pt3[0,1] - pt2[0,1]
    
    #convert the face dlib rect to a boundingbox
    #used for drawing the rectangle around face
    def convert_to_rect(self):
        tl = self.dlib_rect.tl_corner()
        tr = self.dlib_rect.tr_corner()
        bl = self.dlib_rect.bl_corner()
        br = self.dlib_rect.br_corner()
        return (tl.x, tl.y,tr.x - tl.x , br.y-tl.y )

    #draw all 64 landmarks of face
    def draw_all_landmarks(self, img):
        for pt in self.landmarks:
                pos = (pt[0,0], pt[0,1])
                cv2.circle(img, pos, 2, WHITE, -1)

    #grab points of bounding box, as well as dimensions based on passed in points
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

    #helper function for is_smiling function
    def smile_ratio(self):
        props = self.grab_rotated_bbox(self.landmarks[48:55])
        w = props["w"]
        h = props["h"]
        r = (w/self.w)/(h/self.h)
        return r

    def my_is_smiling(self):
        sm_ratio = self.smile_ratio()
        return sm_ratio > 7

    def get_eye_ratios(self):
        l_rbbox = self.grab_rotated_bbox(self.landmarks[36:40])
        r_rbbox = self.grab_rotated_bbox(self.landmarks[42:46])

        lw = l_rbbox["w"]
        lh = l_rbbox["h"]
        rw = r_rbbox["w"]
        rh = r_rbbox["h"]

        l = (lh/self.h)
        r = (rh/self.h)
        return l,r

    #my method to test if eyes are blinking
    def my_eyes_open(self):
        l, r = self.get_eye_ratios()
        return (l+r)/2 > .038

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

    #draw ratio lines around mouth
    def draw_mouth_lines(self, img):
        props = self.grab_rotated_bbox(self.landmarks[48:55])
        coords = props["pts"]
        x,y = coords[0] #bottom left
        x1,y1 = coords[1] #top left
        x2,y2 = coords[2] #top right
        x3,y3 = coords[3] #bottom right
        #draw aspect ratio lines
        cv2.line(img,(x,y),(x1,y1),BLUE,1)
        cv2.line(img,(x1,y1),(x2,y2),BLUE,1)
        
        r = self.smile_ratio()
        self.put_text_img(img, str(round(r,2)), ORANGE, self.landmarks[50])

    #draw ratio lines around all eyes
    def draw_eye_lines(self, img):
        l, r = self.get_eye_ratios()
        new_r = (l+r)/2

        props = self.grab_rotated_bbox(self.landmarks[36:42])
        coords = props["pts"]
        x,y = coords[0] #bottom left
        x1,y1 = coords[1] #top left
        x2,y2 = coords[2] #top right
        x3,y3 = coords[3] #bottom right
        #draw aspect ratio lines
        cv2.line(img,(x,y),(x1,y1),BLUE,1)
        cv2.line(img,(x1,y1),(x2,y2),BLUE,1)

        props = self.grab_rotated_bbox(self.landmarks[42:48])
        coords = props["pts"]
        x,y = coords[0] #bottom left
        x1,y1 = coords[1] #top left
        x2,y2 = coords[2] #top right
        x3,y3 = coords[3] #bottom right
        #draw aspect ratio lines
        cv2.line(img,(x,y),(x1,y1),BLUE,1)
        cv2.line(img,(x1,y1),(x2,y2),BLUE,1)

        self.put_text_img(img, str(round(new_r,3)), ORANGE, self.landmarks[27])

    #put something on the image
    def put_text_img(self, img, text, color, pt):
        x = pt[0,0]
        y = pt[0,1]
        x-=5
        y-=5
        cv2.putText(img,text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2, cv2.LINE_AA)
  
class Detector:
    def __init__(self,option,debugging,save):
        self.detector = dlib.get_frontal_face_detector()
        self.option = option
        self.debugging = debugging
        self.save = save
        #generate the save folder
        self.save_path = "../data/Perfect_photo"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
    
    def savePhoto(self, img):
        num = len(glob.glob(self.save_path+'/*'))
        path = self.save_path+"/" + str(num+1) + ".jpg"
        cv2.imwrite(path, img)

    def get_perfect(self, face, img):
        smiling = True
        eye_open = True
        if self.option == "eyes":
            eye_open = face.my_eyes_open() #get eyes
            eye_color = GREEN if eye_open else RED
            face.draw_eyes(img, eye_color)
        elif self.option == "smile":
            smiling = face.my_is_smiling() #get smile
            box_color = GREEN if smiling else RED
            face.draw_face_bbox(img, box_color)
        elif self.option == "both":
            eye_open = face.my_eyes_open() #get eyes
            eye_color = GREEN if eye_open else RED
            face.draw_eyes(img, eye_color)
            smiling = face.my_is_smiling() #get smile
            box_color = GREEN if smiling else RED
            face.draw_face_bbox(img, box_color)
        return smiling and eye_open

    def show_debug(self, face, img):
        if self.option == "eyes":
            face.draw_eye_lines(img)
        elif self.option == "smile":
            face.draw_mouth_lines(img)
        elif self.option == "both":
            face.draw_eye_lines(img)
            face.draw_mouth_lines(img)

    def perfectPhoto(self, img):
        orig_img = img.copy()
        #convert to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #grab all dlib rects
        dlib_rects =  self.detector(gray, 1) 
        #create face objects out of rects
        faces = [Face(rect,img) for rect in dlib_rects] 
        #boolean if we will save this image
        perfect = True if len(faces) > 0 else False
        #iterate over the image
        for face in faces:
            temp = self.get_perfect(face, img)
            #reassign photo
            if perfect:
                perfect = temp
            #show debug output
            if self.debugging:
                self.show_debug(face,img)
        #should we save this photo?
        if perfect and self.save:
            self.savePhoto(orig_img)
            print("Saving photo")
        return perfect