import glob 
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
    @abstractmethod
    def get_size(self):
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
    
    def get_size(self):
        return int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)),int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))


class FrameList(Frames):
    def __init__(self, source):
        super().__init__(source)
        self.list = glob.glob(self.source+'/*') 
        self.list = [x[x.rfind('/')+1:] for x in self.list]
        self.list = [x[x.rfind('\\')+1:] for x in self.list] #DO THIS FOR WINDOWS
        self.counter = 0
        self.w = -1
        self.h = -1

    #overwrittern function
    def get_frame(self):
        curr_ind = self.counter
        self.counter +=1
        if self.counter >= len(self.list):
            self.finished = True
        name = self.source + '/' + self.list[curr_ind]
        img = cv2.imread(name)
        self.h,self.w = img.shape[:2]
        return self.list[curr_ind], img

    def get_size(self):
        return self.w, self.h