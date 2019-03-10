import argparse
import glob 
import xlsxwriter
import cv2
from Frames import Frames, VideoList, FrameList
import os

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

    #ARRAY FOR PRINTING OUT SHIT
    arr = ["Smile TP: ", "Smile FP: ", "Smile TN: ", "Smile FN: ", \
           "Blink TP: ", "Blink FP: ", "Blink TN: ", "Blink FN: "]
    row = 0
    workbook = xlsxwriter.Workbook('../results/'+folder+'.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write(row, 0, "Image Name")
    worksheet.write(row, 1, "Smiling True Positive")
    worksheet.write(row, 2, "Smiling False Positive")
    worksheet.write(row, 3, "Smiling True Negative")
    worksheet.write(row, 4, "Smiling False Negative")
    worksheet.write(row, 5, "Blinking True Positive")
    worksheet.write(row, 6, "Blinking False Positive")
    worksheet.write(row, 7, "Blinking True Negative")
    worksheet.write(row, 8, "Blinking False Negative")

    #ITERATE OVER ALL IMAGES
    while not images.is_finished():
        name, frame = images.get_frame()
        row+=1
        worksheet.write(row, 0, name)
        col = 1
        cv2.imshow("Feed", frame)
        for a in arr:
            print(a)
            k = cv2.waitKey()
            print(chr(k))
            if k == ord('q'):
                break
            worksheet.write(row, col, chr(k))
            col+=1
        print("moving on to next photo\n")
    workbook.close()
if __name__ == '__main__':
    main()
