import os, sys
import cv2
import numpy as np
import glob
import pandas
from. char_segment import *

##  centre of gravity 
def calc_center(bn_img):
    M = cv2.moments(bn_img)
    try:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    except ZeroDivisionError as e:
        cx = int(bn_img.shape[1]/2)
        cy = int(bn_img.shape[0]/2)
    return cx,cy

def calc_compactness(bn_img):
    assert bn_img.shape[0]>0
    assert bn_img.shape[1]>0
    _img = bn_img.copy()
    _img[_img>0]=1
    cnt = np.sum(_img)
    return cnt/(_img.shape[0]*_img.shape[1])

def calc_fitellipse(bn_rgn_img):
    _img = bn_rgn_img.astype(np.uint8)
    image,contours,hierarchy=cv2.findContours(_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
    axe_l = 0
    axe_s= 0
    for contour in contours:   
        if len(contour) > 5:
            ellipse = cv2.fitEllipse(contour) 
            if axe_l<ellipse[1][0]:
                axe_l=ellipse[1][0]
            if axe_s<ellipse[1][1]:
                axe_s=ellipse[1][1]
    return int(axe_l),int(axe_s)

def read_rects(csv_file):
    df=pandas.read_csv(csv_file,header=None,sep=',',names=["x", "y", "width", "height", "confidence"])

    rects = list()
    for i in range(1,len(df)):
       x = int(df["x"][i])
       y = int(df["y"][i])
       w = int(df["width"][i])
       h = int(df["height"][i])
       rects.append([x,y,w,h])
    return rects

def write_rects(rects,csv_file):
    dp= pandas.DataFrame(rects,columns=["x", "y", "width", "height"])
    dp.to_csv(csv_file)

# scan a line on a direction
def _line_scan_1eft(bn_img,pts,step,max_len,thres):
    cnt = 0.0
    while(True):
        if pts[0][0]<=step or pts[1][0]<=step or cnt>max_len or \
           np.mean(bn_img[pts[0][1]:pts[1][1],pts[0][0]-step:pts[0][0]]/255.0)<thres:  
            break
        pts[0][0]-=step
        pts[1][0]-=step
        cnt += step
    return pts

def _line_scan_right(bn_img,pts,step,max_len,thres):
    cnt = 0
    while(True):
        if pts[0][0]>bn_img.shape[1]-step or pts[1][0]>bn_img.shape[1]-step or cnt>max_len or  \
           np.mean(bn_img[pts[0][1]:pts[1][1],pts[0][0]:pts[0][0]+step]/255.0)<thres:
            break
        pts[0][0]+=step
        pts[1][0]+=step
        cnt += step
    return pts


def _local_threshold(gray_image,rcts):
    bn_image = np.zeros(gray_image.shape)
    rcts = np.array(rcts,dtype=np.int32)
    for rct in rcts:
        _rct = rct.copy()
        ex_len = rct[3]*2
        _rct[0] = max(0,rct[0]-ex_len)
        _rct[1] = max(0,rct[1]-ex_len)
        _rct[2] = min(gray_image.shape[1]-_rct[0]-1,rct[2]+2*ex_len)
        _rct[3] = min(gray_image.shape[0]-_rct[1]-1,rct[3]+2*ex_len)

        roi = gray_image[_rct[1]:_rct[1]+_rct[3],_rct[0]:_rct[0]+_rct[2]]
        blur = cv2.GaussianBlur(roi,(3,3),0)
        ret,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        dst = bn_image[_rct[1]:_rct[1]+_rct[3],_rct[0]:_rct[0]+_rct[2]]
        dst[th==255]=255
         
    return bn_image
 
###
###
def rects_adjust(im,rcts,fore_thres=0.1,extend_max_ratio=0.3,tlh_ratio_thres=0.5,level=0):
    if im.dtype != np.uint8:
        im = im.astype(np.uint8)
    if im.shape[2] != 1:
        gray_image = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    else:
        gray_image = im
    
    bn_img = _local_threshold(gray_image,rcts)
    kernel = np.ones((5,5), np.uint8)
    bn_img = cv2.dilate(bn_img, kernel, iterations=1)
  
    _rcts = np.array(rcts).copy()
    for i in range(len(_rcts)):
        r = _rcts[i].astype(int)
        rgn = bn_img[r[1]:r[1]+r[3],r[0]:r[0]+r[2]].copy()
        cx,cy = calc_center(rgn)
        _rcts[i][0] += (cx-int(r[2]/2))
        _rcts[i][1] += (cy-int(r[3]/2))
        # crop rect outside of image
        _rcts[i][0] = max(0,_rcts[i][0])
        _rcts[i][1] = max(0,_rcts[i][1])
        _rcts[i][0] = min(bn_img.shape[1]-1,_rcts[i][0])
        _rcts[i][1] = min(bn_img.shape[0]-1,_rcts[i][1])

    
    dirc = np.array([[-1,0],[0,-1],[1,0],[0,1]])
    
    rlt_rcts = []
    for i in range(len(_rcts)):
        r = _rcts[i].astype(int)
        _left = np.array([[r[0],r[1]+int(r[3]/4)],[r[0],r[1]+int(r[3]*3/4)]])
        _right = np.array([[r[0]+r[2],r[1]+int(r[3]/4)],[r[0]+r[2],r[1]+int(r[3]*3/4)]])
        #up = np.array([[r[0],r[1]],[r[0]+r[2],r[1]]])
        #bottom = np.array([[r[0],r[1]+r[3]],[r[0]+r[2],r[1]+r[3]]])
        _rl = r.copy()
        left = _line_scan_1eft(bn_img,_left,int(r[3]/3),int(r[3]*extend_max_ratio),fore_thres)
        _rl[2] += r[0]-left[0][0]
        _rl[0] = left[0][0]
        _rr = r.copy()
        right = _line_scan_right(bn_img,_right,int(r[3]/3),int(r[3]*extend_max_ratio),fore_thres)
        #_rr[2] = right[0][0]-r[0]
        r[0] = _rl[0]
        r[2] = right[0][0]-r[0]
        _rcts[i] = r

        _rcts[i][4] = rcts[i][4]  #assign the confidence
        
        ## filter out strange rects according to boundrect
        #cmrcts = _rcts[i].astype(int)      
        #local_rng_w = min(cmrcts[2],cmrcts[3]*2)
        #local_rng_h = cmrcts[3]
        #_rn = bn_img[cmrcts[1]:cmrcts[1]+cmrcts[3],
        #  cmrcts[0]+int(cmrcts[2]/2)-int(local_rng_w/2): \
        #  cmrcts[0]+int(cmrcts[2]/2)+int(local_rng_w/2)]
        #w,h = calc_fitellipse(_rn)
        #_long = max(w,h)
        #_short = min(w,h)
        #if _long>0 and _short>0:
        #    if _short/cmrcts[3]> tlh_ratio_thres:         
     
        ### filter out non-characters
        if level == 1:
            rgn = bn_img[int(_rcts[i][1]):int(_rcts[i][1]+_rcts[i][3]),
              int(_rcts[i][0]):int(_rcts[i][0]+_rcts[i][2])].astype(np.uint8)
            _rct_back = _rcts[i].copy()
            hbound = locate_horizon_bound(rgn)
            if len(hbound) > 0 :
                left_bnd = _rcts[i][0]+hbound[0]
                if 1:#left_bnd <=_left[0][0]:
                    _rct_back[0]+=hbound[0]
                    _rct_back[2]-=hbound[0]
                    
                right_bnd = _rcts[i][0]+hbound[1]
                if 0:#right_bnd >=_right[0][0]:
                    _rct_back[2] = _rcts[i][0]+ hbound[1]-_rct_back[0]
                    print(_rcts[i][0], hbound[1],_rct_back[0])
                _rcts[i] = _rct_back
    
        ## filter out low compactness boxes
        #compactness = calc_compactness(_rn)
        #if compactness > compact_thres:
     
        rlt_rcts.append(_rcts[i].tolist())
    return rlt_rcts
    

def drawboxes(image,rects):
    for i in range(len(rects)):
        cv2.rectangle(image, (rects[i][0],rects[i][1]),  \
            (rects[i][0]+rects[i][2],rects[i][1]+rects[i][3]), (255, 0, 0), 2)


if __name__ == "__main__":
    print('args: image_file_path[option]')
    if len(sys.argv) == 2:
        image_path = sys.argv[1]
    else:
        image_path='/home/gaolin/TextBoxes/data/train_ticket'

    im_names = glob.glob(os.path.join(image_path,"*.jpg"))   
	
    for im_name in im_names: 	
        print(im_name)
        cvs_file_tbx = im_name.replace('.jpg','_tb.csv') 
        im =cv2.imread(im_name, 1)
        
        rcts_tbx = read_rects(cvs_file_tbx)
       # rcts_fus = rects_fusion_simple(rcts_tbx,rcts_ipd,im.shape[:2])
        rcts_adj = rects_adjust(im,rcts_tbx)
       
        im_c = im.copy()
        drawboxes(im_c,rcts_tbx)
        sv_name = im_name.replace('.jpg','_tbx.jpg')
        cv2.imwrite(os.path.join(os.path.dirname(sv_name),
                 'result',os.path.basename(sv_name)),im_c)

        im_c = im.copy()
        drawboxes(im_c,rcts_adj)
        sv_name = im_name.replace('.jpg','_adj.jpg')
        cv2.imwrite(os.path.join(os.path.dirname(sv_name),
                 'result',os.path.basename(sv_name)),im_c)

