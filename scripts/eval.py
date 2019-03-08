import argparse
import glob 
import xlsxwriter
import cv2
from threading import Thread
from abc import ABC, abstractmethod

class Frames(ABC):
    def __init__(self,source):
        if source[-1] == '/':
            source = source[:-1] #remove any ending / just in case
        self.source = source
        self.finished = False
        pass
    
    def is_finished(self):
        return self.finished

    def get_source(self):
        return self.source

    def stop(self):
        self.finished = True

    @abstractmethod
    def get_frame(self):
        pass

class VideoList(Frames):
    def __init__(self,source):
        super().__init__(source)
        self.stream = cv2.VideoCapture(0)
        _,self.frame = self.stream.read()
        self.start()

    #start the thread
    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    #thread continously grabs frames from camera
    def update(self):
        while True:
            if self.is_finished():
                self.stream.release()
                return
            _, self.frame = self.stream.read()

    #overwrittern function
    def get_frame(self):
        return 'webcam', self.frame

class FrameList(Frames):
    def __init__(self, source):
        super().__init__(source)
        self.list = glob.glob(self.source+'/*') 
        self.list = [x[x.rfind('/')+1:] for x in self.list]
        self.counter = 0

    #overwrittern function
    def get_frame(self):
        curr_ind = self.counter
        self.counter +=1
        if self.counter >= len(self.list):
            self.finished = True
        return self.list[curr_ind], cv2.imread(self.source + '/' + self.list[curr_ind])

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

    while not images.is_finished():
        name, frame = images.get_frame()
        cv2.imshow("Feed", frame)
        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'):
            images.stop()

if __name__ == '__main__':
    main()
