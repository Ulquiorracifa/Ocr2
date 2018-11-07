import importlib
import numpy as np
import cv2
import matplotlib.pyplot as pl

from ..core import trans, lineseg

importlib.reload(trans)
importlib.reload(lineseg)

from . import _wireframe_corner
importlib.reload(_wireframe_corner)
from ._wireframe_corner import CornerPointDetect

from . import _wireframe_prelocate
importlib.reload(_wireframe_prelocate)
from ._wireframe_prelocate import PreLocateWireframe as Prelocate

from ..frame import _surface_extract

importlib.reload(_surface_extract)
from ._surface_extract import RelativeBoxPoints

from . import _wireframe_template
importlib.reload(_wireframe_template)
from ._wireframe_template import WireframeTemplateData, WireframeTemplate

import fp.util.visualize

importlib.reload(fp.util.visualize)
            
class Detect(object):
    def __init__(self, debug=False):
        self.detect_corners = CornerPointDetect(debug=debug)
        self.prelocate = Prelocate(debug=debug)
        self.debug = dict() if debug else None
        
    def __call__(self, image):
        '''
        output box'''
        # print('     Wirframe.Detect begin')
        _line_len_th = 0.02 * np.min(image.shape[:2])
        lines = lineseg.detect_lines(image)
        lines = lineseg.remove_tiny_lines(lines, _line_len_th)
        major_lines = lineseg.remove_tiny_lines(lines, 4 * _line_len_th)
        # print('     Wirframe.Detect.lines done')
        points, _ = self.detect_corners(lines)
        major_points, _ = self.detect_corners(major_lines)
        # print('     Wirframe.Detect.points done')
        image_size = image.shape[1], image.shape[0]
        box, init_rectr = self.prelocate(major_points, points, image_size)
        # print('     Wirframe.Detect.box done')

        if self.debug is not None:
            disp = fp.util.visualize.lines(image, lines, color=(255, 120, 0))
            disp = fp.util.visualize.points(disp, points, radius=4, color=(255, 120, 0))
            disp = fp.util.visualize.lines(disp, major_lines, color=(0, 0, 255))
            disp = fp.util.visualize.points(disp, major_points, radius=4, color=(0, 0, 255))
            disp = fp.util.visualize.box(disp, box, color=(0, 255, 255))
            self.debug['result'] = disp
            pl.figure(figsize=(13, 13))
            pl.imshow(disp)
        
        return points, box, init_rectr
    
def visualize(image, points):
    imx = cv2.merge((image, image, image))
    for x, y in points:
        x = int(round(x))
        y = int(round(y))
        cv2.circle(imx, (x, y), 3, (255, 0, 0), thickness=1)
    return imx


class Extract(object):
    def __init__(self, relative_surface, relative_head, relative_tail, debug=False):
        self.extract_surface = RelativeBoxPoints(relative_surface)
        self.extract_head = RelativeBoxPoints(relative_head)
        self.extract_tail = RelativeBoxPoints(relative_tail)
        self.debug = dict() if debug else None

    def __call__(self, image, wireframe_box):
        '''
        output a standard image
        '''
        head_pix = self._pixel_mean(self.extract_head, image, wireframe_box)
        tail_pix = self._pixel_mean(self.extract_tail, image, wireframe_box)

        if self.debug is not None:
            print('head_pix', head_pix)
            print('tail_pix', tail_pix)
        if head_pix < tail_pix:
            wireframe_box = wireframe_box[0], wireframe_box[1], wireframe_box[2] + 180.
        surface_points = self.extract_surface(wireframe_box)
        surface_image = trans.deskew(image, surface_points)
        return surface_points, surface_image

    def _pixel_mean(self, points_func, image, box):
        _points = points_func(box)
        _image = trans.deskew(image, _points)
        _image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
        _image = cv2.Laplacian(_image, cv2.CV_16S)
        _image = np.absolute(_image).astype(np.uint8)
        # print('image minmax diff:', np.max(_image) - np.min(_image))
        # _, _image = cv2.threshold(_image, 20, 255, cv2.THRESH_OTSU)
        # unlock this to test
        # pl.figure()
        # pl.imshow(_image, 'gray')

        return np.mean(_image)
