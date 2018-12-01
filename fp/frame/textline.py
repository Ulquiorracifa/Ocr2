import os
import importlib
import numpy as np
import cv2

import fp.config
importlib.reload(fp.config)

from . import _textline_simple_detect
importlib.reload(_textline_simple_detect)
from ._textline_simple_detect import TextlineSimpleDetect

from . import _textline_lenet_classify
importlib.reload(_textline_lenet_classify)
from ._textline_lenet_classify import TextlineLenetClassify

from ..util import check
importlib.reload(check)

from ..util import machine
importlib.reload(machine)

from ..util import data
importlib.reload(data)

if not machine.is_('s60'):  # currently, only 11-server and dgx provides Caffe
    from ..TextBoxes import detect_textline
    importlib.reload(detect_textline)

    TextlineTextBoxesDetect = detect_textline.TextBoxesDetect

class Detect(object):
    '''Textline Detect'''
    default_pars_simple = dict()
    default_pars_textboxes = dict()

    def __init__(self, method='simple', pars={}, debug=False):
        self.method = method
        if method == 'simple':
            self.detect = TextlineSimpleDetect(**pars, debug=debug)
        elif method == 'textboxes':
            self.detect = TextlineTextBoxesDetect(**pars)
            # self.detect = TextlineTextBoxesDetect(**pars, debug=debug)
        else:
            raise NotImplemented

    def __call__(self, image, rois=[]):
        '''
        return list of rects'''
        if self.method == 'simple':
            assert check.valid_image(image, colored=0)
        elif self.method == 'textboxes':
            assert check.valid_image(image, colored=1)
        else:
            raise NotImplemented

        # prepare data for caffe
        if self.method == 'textboxes' or self.method == 'textboxes_gpu':
            image = data.make_caffe_data(image)

        # print('#### Image shape before textline: ', image.shape)
        result = self.detect(image, rois=rois)
        if len(result) == 0:
            return None
        result = np.array(result)
        if result.dtype is not np.int64:
            result = np.round(result).astype(np.int64)
        if result.shape[1] > 4:
            result = result[:, :4]

        # cut the rect-part that is outside of image
        H, W = image.shape[:2]
        for i in range(len(result)):
            x0, y0, w, h = result[i]
            x1, y1 = x0 + w, y0 + h
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x1, W), min(y1, H)
            result[i] = np.array([x0, y0, x1 - x0, y1 - y0])
            
        # remove invalid rects
        result = result[np.bitwise_and(result[:, 2] > 0, result[:, 3] > 0)]
        return result

    
class Classify(object):
    '''Textline Detect'''
    default_pars_lenet = dict(weight_file=fp.config.TEXTLINE_CLASSIFY_LENET_WEIGHT)

    def __init__(self, method='lenet', pars=default_pars_lenet):
        if method == 'lenet':
            self.classify = TextlineLenetClassify(**pars)
        else:
            raise NotImplemented

    def __call__(self, image, rects):
        check.valid_image(image, colored=0)
        check.valid_rects(rects)
        return self.classify(image, rects)