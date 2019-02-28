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

def normalized_a_r(cnt,fw,fh):
    x,y,w,h = cv2.boundingRect(cnt)
    #print("FW: {}, FH: {}, w: {}, h: {} ".format(fw,fh,w,h))
    print("w/fw: {}, h/fh: {}".format(float(Fraction(w,fw)), float(Fraction(h,fh))))
    aspect_ratio = (float(w)/fw)/(float(h)/fh)
    print("A_R: {} \n".format(aspect_ratio))
    return aspect_ratio

def draw_rect(img, x,y,w,h,color):
    cv2.rectangle(img, (x, y), (x + w, y + h),color, 2)



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

    def draw_smile_landmarks(self,img):
        fts = self.landmarks[48:68]
        for pt in fts:
            pos = (pt[0,0], pt[0,1])
            cv2.circle(img, pos, 2, (0, 255, 255), -1)

    def is_smiling(self,img):
        (x,y,w,h) = self.convert_to_rect()
        points = self.landmarks[48:55]
        upper_lip = np.array(points, dtype=np.int32)
        a_r = normalized_a_r(upper_lip, w, h)
        self.draw_smile_line(img)
        if a_r > 10:
            draw_rect(img,x,y,w,h,GREEN)
        else:
            draw_rect(img,x,y,w,h,RED)


    def draw_smile_line(self, img):
        (x,y,w,h) = self.convert_to_rect()
        points = self.landmarks[48:55]
        contours = [np.array(points, dtype=np.int32)]
        for cnt in contours:
            cv2.drawContours(img,[cnt],0,(0,255,0),2)
            #print(area)

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