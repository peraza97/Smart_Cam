import cv2
import argparse
from Frames import Frames, VideoList, FrameList
from Detector import Detector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera", help="using camera feed", action='store_true')
    parser.add_argument("-p", "--path", help="path to images")
    parser.add_argument("-d", "--debugging", help="options : both, eyes, face")
    args = parser.parse_args()

    time = 1
    if args.camera:
        images = VideoList('Camera')
    elif args.path:
        time = 2500
        images = FrameList(args.path)
    else:
        raise Exception('must pass either feed or images') 

    detector = Detector(debugging=args.debugging)

    while not images.is_finished():
        #grab a frame
        _, frame = images.get_frame()
        #detect perfect photo
        detector.perfectPhoto(frame)
        #show the photo
        cv2.imshow("Feed", frame)
        k = cv2.waitKey(time)
        if k & 0xFF == ord('q'):
            images.stop()

if __name__ == '__main__':
    main()