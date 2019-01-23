import numpy as np
import os
os.environ['GLOG_minloglevel']= '3'

import cv2
import glob
import pandas
import sys
import datetime
import caffe

_version_type_ = 'release'

Max_Blob_Width = 1300
Max_Blob_Height = 1300

if _version_type_ == 'debug':
    from nms import nms
    from util import timewatch,drawboxes, createborder
    from caffe_config import model_config, selected_rois_keys
    from rects_adjust import rects_adjust
    if True:
        caffe.set_mode_gpu()
        caffe.set_device(2)
    else:
        caffe.set_mode_cpu()
else:
    from .nms import nms
    from .util import timewatch,drawboxes,createborder
    from .caffe_config import model_config,selected_rois_keys
    from .rects_adjust import rects_adjust
    from .. import config
    if config.TEXTLINE_DETECT_USE_CUDA is True:
        caffe.set_mode_gpu()
        caffe.set_device(2)
    else:
        caffe.set_mode_cpu()

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
## load caffe package
from .. import util
if util.machine.is_('dgx'):
    #pwd = os.getcwd()
    caffe_root = FILE_DIR # Make sure this file is on the caffe root path
    #os.chdir(caffe_root)
    #sys.path.insert(0, os.path.join(caffe_root, 'python'))
    sys.path.append(os.path.join(caffe_root, 'python'))
elif util.machine.is_('s60'):
    #caffe_root = None # @todo FIX THIS!
    pass
else:
    raise NotImplemented
    


class _Detect(object):
    def __init__(self):
        pass

    def __call__(self, image):
        raise NotImplemented

        
class TextBoxesDetect(_Detect):
    def __init__(self,   \
        model_def = FILE_DIR +'/'+'models/fapiao.prototxt',  \
        model_weights = FILE_DIR +'/'+'models/fapiao.caffemodel', \
        scales= ((170,420),(340,840),(680,840)), \
        confidence_thres = 0.4): # 

        if True:
            caffe.set_device(2)
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()        

        tim = timewatch.start_new()
        self.model_def = model_def
        self.model_weights =  model_weights
        #print(self.model_def,self.model_weights)
        self.net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
        self.scales = scales
        self.confidence_thres = confidence_thres
        self.predictions = []
        keys = list(model_config.keys())
        self.model_config_key = None#keys[0]
        self.postprocess_level = 0
        self.zoomfactor = 1.0
        #print("load model parameters: ", tim.get_elapsed_seconds(), " seconds")

    def __call__(self, image, rois=dict(), model_key='700x700-3'):
        tim = timewatch.start_new()
        if max(image.shape[0],image.shape[1])>2000:  # scaling down the scales
            self.zoomfactor = 0.5            
        if model_key is not None and model_key!=self.model_config_key:
            self.load_model_config(model_key)
            self.model_config_key = model_key
        roi_num = len(rois)
        if roi_num > 0:
            predictions = []
            for _key in rois.keys():
                if _key in selected_rois_keys or _version_type_ == 'debug':
                    _roi = rois[_key]
                    print(_key)
                    if _key=='header0' or _key=='tax_free':  ### the item is crossed with the wire frame
                        _roi = list(_roi)
                        _roi[2] += _roi[2]/6
                    if _key=='header0':
                        self.set_postprocess(level=1)  ## remove the 2D-code partly in the textline
                    else:
                        self.set_postprocess(level=0)
                    roi = [int(x) for x in _roi]
                    prediction = self.detect_roi(image,roi)
                    predictions+=prediction
            self.predictions = predictions
        else:
            self.predictions = self.detect(image)
        print("Textline Extraction Done in : ", tim.get_elapsed_seconds(), " seconds")
        #self.draw_boxes_tofile(image,'/home/tangpeng/temp/'+str(tim.get_time())+'.jpg') 
        return self.predictions

    def draw_boxes_tofile(self, image, filepath):
        sim = (image*255.0).astype(np.uint8)
        drawboxes(sim,self.predictions)
        cv2.imwrite(filepath,sim)
        print('Text Boxes are drawed in ',filepath)

    '''
    roi = [x,y,w,h]
    '''
    def detect_roi(self,image,roi,extend=True):
        assert roi[1]<roi[1]+roi[3]
        assert roi[0]<roi[0]+roi[2] 
        im = image[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2],:]

        bot_exd = 80
        if extend:
            im = createborder(im,bottom=bot_exd,val=1)
        prediction = self.detect(im)
        if extend:
           _pred = []
           for i in range(len(prediction)):
               if prediction[i][1]+prediction[i][3]-10 < im.shape[0]-bot_exd:
                   _pred.append(prediction[i])
           prediction = _pred

        for i in range(len(prediction)):
            prediction[i][0]+=roi[0]
            prediction[i][1]+=roi[1]
        return prediction
    
    def save_predictions_to_cvs(self,cvs_file=None):
        if cvs_file is not None:
            df = pandas.DataFrame(self.predictions, columns=["x", "y", "width", "height", "confidence"])
            df.to_csv(cvs_file, index=False)

    def set_net_model(self, model_def, model_weights):
        self.model_def = model_def
        self.model_weights =  model_weights
        self.net = caffe.Net(model_def, model_weights, caffe.TEST)    

    def load_model_config(self,config_name):
        self.set_net_model(FILE_DIR + '/'+ model_config[config_name]['model_def'], \
                           FILE_DIR + '/'+ model_config[config_name]['model_weights'])
        self.scales = model_config[config_name]['scales']    

    def set_postprocess(self,level=0):
        self.postprocess_level = level

    def detect(self, image):
        #image=caffe.io.load_image(image_file)
        image_height,image_width,channels=image.shape
        print('image(roi)',image.shape)
        _rlts=[]
        for scale in self.scales:
            if type(scale[0]) is float:
                image_resize_height = int(image_height * scale[0] * self.zoomfactor) #  height
                image_resize_width = int(image_width * scale[1] * self.zoomfactor)  #  width
            else:
                image_resize_height = scale[0]  ##0 height
                image_resize_width =  scale[1]   ##1 width
            if image_resize_width > Max_Blob_Width:
                image_resize_width = Max_Blob_Width
            if image_resize_height > Max_Blob_Height:
                image_resize_height = Max_Blob_Height
            
            transformer = caffe.io.Transformer({'data': (1,3,image_resize_height,image_resize_width)})
            transformer.set_transpose('data', (2, 0, 1))
            transformer.set_mean('data', np.array([104,117,123])) # mean pixel
            transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
            transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
        
            self.net.blobs['data'].reshape(1,3,image_resize_height,image_resize_width)        
            transformed_image = transformer.preprocess('data', image)
            self.net.blobs['data'].data[...] = transformed_image
            # Forward pass.
            detections = self.net.forward()['detection_out']
        
            # Parse the outputs.
            det_label = detections[0,0,:,1]
            det_conf = detections[0,0,:,2]
            det_xmin = detections[0,0,:,3]
            det_ymin = detections[0,0,:,4]
            det_xmax = detections[0,0,:,5]
            det_ymax = detections[0,0,:,6]
            top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence_thres]
            top_conf = det_conf[top_indices]
            top_xmin = det_xmin[top_indices]
            top_ymin = det_ymin[top_indices]
            top_xmax = det_xmax[top_indices]
            top_ymax = det_ymax[top_indices]

            for i in range(top_conf.shape[0]):
                xmin = int(round(top_xmin[i] * image.shape[1]))
                ymin = int(round(top_ymin[i] * image.shape[0]))
                xmax = int(round(top_xmax[i] * image.shape[1]))
                ymax = int(round(top_ymax[i] * image.shape[0]))
                xmin = max(1,xmin)
                ymin = max(1,ymin)
                xmax = min(image.shape[1]-1, xmax)
                ymax = min(image.shape[0]-1, ymax)
                score = top_conf[i]
                dt_result=[xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax,score]
                _rlts.append(dt_result)

        _rlts = sorted(_rlts, key=lambda x:-float(x[8])) 
        nms_flag = nms(_rlts,0.1)
        det_results = []
        for k,dt in enumerate(_rlts):
            if nms_flag[k]:
                xmin = dt[0]
                ymin = dt[1]
                xmax = dt[2]
                ymax = dt[5]
                conf = dt[8]
                det_results.append([xmin, ymin, xmax-xmin+1, ymax-ymin+1,conf])
        
        if self.postprocess_level>=0:
            #adjust the result rects 
            predictions = rects_adjust((image*255.0).astype(np.uint8),det_results,
               extend_max_ratio=1.0,level=self.postprocess_level)
        else:
            predictions = det_results

        return predictions  # [[(x,y,width,height),confidence],......]


if __name__ == "__main__":
    print('args: image_dir_path[option]')
    if len(sys.argv) == 2:
        image_path = sys.argv[1]
    else:
        image_path='/home/gaolin/data/fapiao'
    im_names = glob.glob(os.path.join(image_path,"*.jpg"))   

    detector = TextBoxesDetect()
    for im_name in im_names:
        cvs_file = im_name.replace('.jpg','.csv')
        tim = timewatch.start_new()
        im =caffe.io.load_image(im_name)
        #im =cv2.imread(im_name,1)
        detector(im)
        #detector.detect_to_cvs(im, cvs_file)
        print(im_name)
        print("Done in : ", tim.get_elapsed_seconds(), " seconds")
       


