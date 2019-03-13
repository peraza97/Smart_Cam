import cv2
import argparse
from Frames import Frames, VideoList, FrameList
from Detector import Detector
import glob
import os

def main():
    images = VideoList('Camera')

    save_path = '../data/Webcam'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    num = len(glob.glob(save_path+'/*'))
    
    while not images.is_finished():
        name, frame = images.get_frame()

        cv2.imshow("Feed", frame)
        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'):
            images.stop()
        elif k & 0xFF == ord('s'):
            gen_path = save_path+"/" + str(num) + ".jpg"
            cv2.imwrite(gen_path, frame)
            num +=1

if __name__ == '__main__':
    main()