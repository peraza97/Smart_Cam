import cv2
import argparse
from Frames import Frames, VideoList, FrameList
from Detector import Detector
import glob
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path to images", required=True)
    args = parser.parse_args()

    images = VideoList('Camera')

    path = args.path
    if path[-1] == '/':
        path = path[:-1] #remove any ending / just in case

    if not os.path.exists(path):
        os.makedirs(path)
    num = len(glob.glob(path+'/*'))
    while not images.is_finished():
        name, frame = images.get_frame()

        cv2.imshow("Feed", frame)
        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'):
            images.stop()
        elif k & 0xFF == ord('s'):
            save_path = path+"/" + str(num) + ".jpg"
            cv2.imwrite(save_path, frame)
            num +=1

if __name__ == '__main__':
    main()