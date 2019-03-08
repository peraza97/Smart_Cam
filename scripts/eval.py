import argparse
import glob 
import xlsxwriter
import cv2
from Frames import Frames, VideoList, FrameList

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera", help="using camera feed", action='store_true')
    parser.add_argument("-p", "--path", help="path to images")
    args = parser.parse_args()

    if args.camera:
        images = VideoList('Camera')
    elif args.path:
        images = FrameList(args.path)
    else:
        raise Exception('must pass either feed or images') 
    
    DISPLAY = False
    source = images.get_source()
    folder = source.split('/')[-1]
    workbook = xlsxwriter.Workbook('../results/'+folder+'.xlsx')
    worksheet = workbook.add_worksheet()
    row = 1

    while not images.is_finished():
        name, frame = images.get_frame()
        worksheet.write(row, 0, name)
        worksheet.write(row, 1, 1)
        row+=1
        if DISPLAY:
            cv2.imshow("Feed", frame)
            k = cv2.waitKey(1)
            if k & 0xFF == ord('q'):
                images.stop()

    workbook.close()
if __name__ == '__main__':
    main()
