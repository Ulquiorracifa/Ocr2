import os
import sys
import datetime
import cv2
import numpy as np

class timewatch(object):

    def start(self):
        self.start_time = timewatch.get_time()

    def get_elapsed_time(self):
        current_time = timewatch.get_time()
        res = current_time - self.start_time
        return res

    def get_elapsed_seconds(self):
        elapsed_time = self.get_elapsed_time()
        res = elapsed_time.total_seconds()
        return res

    @staticmethod
    def get_time():
        res = datetime.datetime.now()
        return res

    @staticmethod
    def start_new():
        res = timewatch()
        res.start()
        return res

def createborder(image,left=0,right=0,top=0,bottom=0,val=255):
    im = image.copy()
    if image.shape[2] != 1:
        extend = np.ones((top+bottom+image.shape[0],left+right+image.shape[1],3))*val
        extend[top:top+image.shape[0],left:left+image.shape[1]] = image
    else:
        extend = np.ones((top+bottom+image.shape[0],left+right+image.shape[1]))*val
        extend[top:top+image.shape[0],left:left+image.shape[1],:] = image
    return extend
    

def drawboxes(image,rects,color=(255,0,0),show_index=False,show_conf=False):
    for i in range(len(rects)):
        cv2.rectangle(image, (int(rects[i][0]),int(rects[i][1])),  \
            (int(rects[i][0])+int(rects[i][2]),int(rects[i][1])+int(rects[i][3])), color, 1)
        
        if show_index:
            cv2.putText(image,"{}".format(i),(int(rects[i][0]),int(rects[i][1])),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),5)
        if show_conf:
            cv2.putText(image,"%.2f"%(rects[i][4]),(int(rects[i][0]+rects[i][3]),int(rects[i][1])),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),3)

