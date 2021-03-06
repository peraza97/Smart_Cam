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
    parser.add_argument("-o", "--option", help="options : both, eyes, smile",required=True)
    parser.add_argument("-d", "--display", help="do you want to display",action='store_true')
    args = parser.parse_args()

    if args.camera:
        images = VideoList('Camera')
    elif args.path:
        images = FrameList(args.path)
    else:
        raise Exception('must pass either feed or images') 
    
    source = images.get_source()
    folder = source.split('/')[-1]

    save = '../data/Results'

    if not os.path.exists(save):
        os.makedirs(save)

    detector = Detector(option=args.option, debugging=False, save=False)

    row = 0
    workbook = xlsxwriter.Workbook(save + '/' + folder + '.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write(row, 0, "Image Name")
    worksheet.write(row, 1, "Results")

    t = 0
    f = 0
    #ITERATE OVER ALL IMAGES
    while not images.is_finished():
        name, frame = images.get_frame()
        value = detector.perfectPhoto(frame)
        if args.display:
            frame = cv2.resize(frame,(400,400), interpolation=cv2.INTER_AREA)
            cv2.imshow("Feed", frame)
            k = cv2.waitKey(1000)
            if k & 0xFF == ord('q'):
                images.stop()
        row+=1
        worksheet.write(row, 0, name)
        worksheet.write(row,1, value*1)
        if value:
            t+=1
        else:
            f+=1
    
    last_row = row + 1
    worksheet.write(last_row+1,0, "Predicted Yes")
    worksheet.write(last_row+1,1, "=COUNTIF(B2:B{},1)".format(last_row) )
    worksheet.write(last_row+2 ,0, "Predicted No")
    worksheet.write(last_row+2,1, "=COUNTIF(B2:B{},0)".format(last_row) )
    workbook.close()
    print("Predicted Yes: {}, Predicted No: {}".format(t,f))
   
if __name__ == '__main__':
    main()
