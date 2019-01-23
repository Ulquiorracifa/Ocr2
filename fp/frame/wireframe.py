import importlib
import numpy as np
import cv2
#import matplotlib.pyplot as pl
import time

from .. import config
from ..core import trans, lineseg
from ..frame import _surface_extract
from ..util import visualize
from . import _wireframe_corner
from . import _wireframe_prelocate
from . import _wireframe_template

if not config.RELEASE:
    importlib.reload(trans)
    importlib.reload(lineseg)
    importlib.reload(visualize)
    importlib.reload(_wireframe_corner)
    importlib.reload(_surface_extract)
    importlib.reload(_wireframe_prelocate)
    importlib.reload(_wireframe_template)

from ._wireframe_corner import CornerPointDetect
from ._surface_extract import RelativeBoxPoints
from ._wireframe_template import WireframeTemplateData, WireframeTemplate
from ._wireframe_prelocate import PreLocateWireframe as Prelocate

def _image_border_point(point, image_size, ratio=0.02):
    '''
    args
      ratio: should less than 0.05
    '''
    x, y = point
    w, h = image_size
    dx = min(abs(x), abs(x - w + 1))
    dy = min(abs(y), abs(y - h + 1))
    th = min(w, h) * ratio
    return dx < th or dy < th
            
class Detect(object):
    def __init__(self, debug=False):
        self.detect_corners = CornerPointDetect(debug=debug)
        self.prelocate = Prelocate(debug=debug)
        self.debug = dict() if debug else None
        
    def __call__(self, image):
        '''
        output box'''
        #print('     Wirframe.Detect begin')
        
        ################
        # detect lines
        if self.debug is not None:
            _t_ = time.time()
        image_size = image.shape[1], image.shape[0]
        _line_len_th = 0.05 * np.min(image.shape[:2])
        lines = lineseg.detect_lines(image)
        lines = lineseg.remove_tiny_lines(lines, _line_len_th)
        major_lines = lineseg.remove_tiny_lines(lines, 1.2 * _line_len_th)
        #print('     Wirframe.Detect.lines done')
        if self.debug is not None:
            _t__ = time.time()
            print('* Detect lines ellapsed {} s.'.format(_t__ - _t_))
            _t_ = _t__
        
        ################
        # detect points
        points, _ = self.detect_corners(lines)
        points = np.array([p for p in points 
                           if not _image_border_point(p, image_size)])
        major_points, _ = self.detect_corners(major_lines)
        # remove image corner point, which is mostly mis-detected
        major_points = np.array([p for p in major_points 
                                 if not _image_border_point(p, image_size)])
        #print(major_points)
        #print('     Wirframe.Detect.points done')
        if self.debug is not None:
            _t__ = time.time()
            print('* Detect points ellapsed {} s.'.format(_t__ - _t_))
            _t_ = _t__
        
        ############
        # prelocate
        box, init_rectr = self.prelocate(major_points, points, image_size)
        #print('     Wirframe.Detect.box done')
        
        if self.debug is not None:
            _t__ = time.time()
            print('* Prelocate ellapsed {} s.'.format(_t__ - _t_))
            _t_ = _t__
        
        if self.debug is not None:
            disp = visualize.box(image, box, color=(0,255,255))
            disp = visualize.lines(disp, lines, color=(255,120,0))
            disp = visualize.points(disp, points, radius=9, color=(255,120,0))
            disp = visualize.lines(disp, major_lines, color=(0,0,255))
            disp = visualize.points(disp, major_points, radius=6, color=(0,0,255))
            self.debug['result'] = disp
            #pl.figure(figsize=(13,13))
            #pl.imshow(disp)
        
        return points, box, init_rectr
    
def visualize_points(image, points):
    imx = cv2.merge((image, image, image))
    for x, y in points:
        x = int(round(x))
        y = int(round(y))
        cv2.circle(imx, (x, y), 3, (255,0,0), thickness=1)
    return imx

class Extract(object):
    def __init__(self, relative_surface, relative_head, relative_tail, debug=False):
        self.relative_rect = relative_surface
        self.extract_surface = RelativeBoxPoints(relative_surface)
        self.extract_head = RelativeBoxPoints(relative_head)
        self.extract_tail = RelativeBoxPoints(relative_tail)
        self.debug = dict() if debug else None
        
    def __call__(self, image, wireframe_box):
        '''
        output a standard image
        '''
        
        # === check if need to rotate 180 ===
        # because a stamp in the header center, the pixel mean should be different
        head_pix = self._pixel_mean(self.extract_head, image, wireframe_box)
        tail_pix = self._pixel_mean(self.extract_tail, image, wireframe_box)
        if self.debug is not None:
            print('head_pix', head_pix)
            print('tail_pix', tail_pix)
        if head_pix < tail_pix:
            # angle + 180, to rotate
            wireframe_box = wireframe_box[0], wireframe_box[1], wireframe_box[2]+180.
        
        # predict points and extract image
        surface_points = self.extract_surface(wireframe_box)
        surface_image = trans.deskew(image, surface_points)
        return surface_points, surface_image
    
    def _pixel_mean(self, points_func, image, box):
        _points = points_func(box)
        _image = trans.deskew(image, _points)
        _image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
        _image = cv2.Laplacian(_image, cv2.CV_16S)
        _image = np.absolute(_image).astype(np.uint8)
        #print('image minmax diff:', np.max(_image) - np.min(_image))
        #_, _image = cv2.threshold(_image, 20, 255, cv2.THRESH_OTSU)
        # unlock this to test
        #pl.figure()
        #pl.imshow(_image, 'gray')
        return np.mean(_image)
    
    def rectr(self):
        rx, ry, rw, rh = self.relative_rect
        dx = -rx / rw
        dy = -ry / rh
        dw = 1.0 / rw
        dh = 1.0 / rh
        return dx, dy, dw, dh