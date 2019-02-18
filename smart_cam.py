import cv2
import dlib
import numpy as np

face_model = cv2.CascadeClassifier('models/front_face.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shapes_predict.dat")

def grab_faces(img,dlib_m=False):
    if dlib_m == False:
        return face_model.detectMultiScale(img, 1.3, 5)
    else:
        return detector(img, 1)

def convert_to_rect(dlib_rect):
            pt = dlib_rect.tl_corner()
            pt2 = dlib_rect.br_corner()
            return (pt.x,pt.y, pt2.x-pt.x,pt2.y-pt.y)


def extract_landmarks(img,faces, dlib_m=False):
    my_dict = []
    if dlib_m == False:
        for (x,y,w,h) in faces:
            rect = dlib.rectangle(x,y,x+w,y+h)
            landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rect).parts()])
            my_dict.append({"rect": (x,y,w,h),"landmarks": landmarks})
    else:
        for rect in faces:
            (x,y,w,h) = convert_to_rect(rect)
            landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rect).parts()])
            my_dict.append({"rect": (x,y,w,h),"landmarks": landmarks})
    return my_dict

def draw_features(img, landmarks):

    for obj in landmarks:
        (x,y,w,h) = obj["rect"]
        points = obj["landmarks"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for pt in points:
            pos = (pt[0,0], pt[0,1])
            cv2.circle(img, pos, 2, (0, 255, 255), -1)

def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read();
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces= grab_faces(gray,1)
        fts = extract_landmarks(gray, faces,1)
        draw_features(frame, fts)  

        cv2.imshow("Feed", frame)
        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'):
            break 
        elif k & 0xFF == ord('s'):
            cv2.imwrite( "./photos/detectedLandmarks.jpg", frame);

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()