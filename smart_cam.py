import cv2
import dlib
import numpy as np
from threading import Thread
from scipy.spatial import distance as dist


#CREATE DETECTOR
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shapes_predict.dat")
smile_model = cv2.CascadeClassifier("models/smile.xml")

def grab_faces(img):
        return detector(img, 1)

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

    def detect_smile(self):
        fts = self.landmarks[48:68]

        #0-67 is my list
        mouth_left = self.landmarks[48]
        mouth_right = self.landmarks[54]

        left_v = self.landmarks[50]
        left_d = self.landmarks[58]

        mid_v = self.landmarks[51]
        mid_d = self.landmarks[57]

        right_v = self.landmarks[52]
        right_d = self.landmarks[56]

        ratio = dist.euclidean(left_v,left_d) + dist.euclidean(mid_v,mid_d) + dist.euclidean(right_v,right_d)
        ratio = ratio/(3 * dist.euclidean(mouth_left,mouth_right))
        print(ratio)

        return ratio > .38 or ratio < .25
        ##do our shit

    def draw_face(self,img):
        (x,y,w,h) = self.convert_to_rect()
        if self.detect_smile():
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    def draw_landmarks(self, img):
        for pt in self.landmarks:
            pos = (pt[0,0], pt[0,1])
            cv2.circle(img, pos, 2, (0, 255, 255), -1)

    def draw_smile_landmarks(self,img):
        fts = self.landmarks[48:68]
        for pt in fts:
            pos = (pt[0,0], pt[0,1])
            cv2.circle(img, pos, 2, (0, 255, 255), -1)

    def draw_smile_harr(self,img):
        (x,y,w,h) = self.convert_to_rect()
        roi_gray = img[y:y+h,x:x+w]

        smile = smile_model.detectMultiScale(
            roi_gray,
            scaleFactor=1.7,
            minNeighbors=22,
            minSize=(25, 25),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for (x1, y1, w1, h1) in smile:
            cv2.rectangle(roi_gray, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 1)
        cv2.imshow("ROI", roi_gray)


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
        try:
            for face in faces:
                face.draw_smile_harr(frame)
        except:
            print("Unexpected error:", sys.exc_info()[0])

        #cv2.imshow("Feed", frame)
        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'):
            c.stop()
            break 
        elif k & 0xFF == ord('s'):
            cv2.imwrite( "./photos/detectedSmile.jpg", frame);

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()