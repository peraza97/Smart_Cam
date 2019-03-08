import cv2
import argparse
from Frames import Frames, VideoList, FrameList
from Detector import Detector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera", help="using camera feed", action='store_true')
    parser.add_argument("-p", "--path", help="path to images")
    parser.add_argument("-d", "--debugging", help="are we debugging output", action='store_true')
    args = parser.parse_args()

    if args.camera:
        images = VideoList('Camera')
    elif args.path:
        images = FrameList(args.path)
    else:
        raise Exception('must pass either feed or images') 

    detector = Detector(debugging=args.debugging)

    while not images.is_finished():
        name, frame = images.get_frame()
        if detector.perfectPhoto(frame):
            print("Perfect photo")

        cv2.imshow("Feed", frame)
        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'):
            images.stop()

if __name__ == '__main__':
    main()