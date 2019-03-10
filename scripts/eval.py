import argparse
import glob 
import xlsxwriter
import cv2
from Frames import Frames, VideoList, FrameList
import os
from Detector import Detector

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
    
    source = images.get_source()
    folder = source.split('/')[-1]

    if not os.path.exists('../results'):
        os.makedirs('../results')

    detector = Detector(debugging="both")

    #ARRAY FOR PRINTING OUT SHIT
    arr = ["Smile TP", "Smile FP", "Smile TN", "Smile FN", \
           "Blink TP", "Blink FP", "Blink TN", "Blink FN"]
    row = 0
    workbook = xlsxwriter.Workbook('../results/'+folder+'.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write(row, 0, "Image Name")
    #fill out names of xlsx sheet
    for i, a in enumerate(arr):
        worksheet.write(row, i+1, arr[i])

    #ITERATE OVER ALL IMAGES
    while not images.is_finished():
        name, frame = images.get_frame()
        detector.perfectPhoto(frame)
        row+=1
        worksheet.write(row, 0, name)
        cv2.imshow("Feed", frame)
        #collect data for current image
        for i, a in enumerate(arr):
            print(a+ ": ")
            k = cv2.waitKey()
            print(chr(k))
            if chr(k) == 'q':
                images.stop()
                break
            worksheet.write(row, i+1, chr(k))
        print("moving on to next photo\n")
    
    workbook.close()
   
if __name__ == '__main__':
    main()
