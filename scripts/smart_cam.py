import cv2
import argparse
from Frames import Frames, VideoList, FrameList
from Detector import Detector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera", help="using camera feed", action='store_true')
    parser.add_argument("-p", "--path", help="path to images")
    parser.add_argument("-o", "--option", help="options : both, eyes, smile",required=True)
    parser.add_argument("-d", "--debugging", help="show debugging steps", action='store_true')
    parser.add_argument("-s", "--save", help="save perfect photos", action='store_true')
    args = parser.parse_args()

    time = 1
    if args.camera:
        images = VideoList('Camera')
    elif args.path:
        time = 1000
        images = FrameList(args.path)
    else:
        raise Exception('must pass either feed or images') 

    detector = Detector(option=args.option, debugging=args.debugging, save=args.save)

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