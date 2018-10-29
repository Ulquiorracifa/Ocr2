import importlib
import numpy as np
import cv2

from . import _wireframe_corner

importlib.reload(_wireframe_corner)
from ._wireframe_corner import LineSegDetect, CornerPointDetect

from . import _wireframe_prelocate

importlib.reload(_wireframe_prelocate)
from ._wireframe_prelocate import PreLocateWireframe as Prelocate

from ..frame import _wireframe_deskew

importlib.reload(_wireframe_deskew)
from ._wireframe_deskew import WireframeDeskew as Deskew

from . import _wireframe_template

importlib.reload(_wireframe_template)
from ._wireframe_template import WireframeTemplateData, WireframeTemplate
            
class Detect(object):
    def __init__(self, debug=False):
        self.detect_lines = LineSegDetect()
        self.detect_corners = CornerPointDetect()
        self.prelocate = Prelocate(debug=debug)
        
    def __call__(self, image):
        lines = self.detect_lines(image)
        points, lineids = self.detect_corners(lines)
        # rects = find_rects(points)
        # print(points)
        image_size = image.shape[1], image.shape[0]
        box, init_rectr = self.prelocate(points, image_size)

        return points, box, init_rectr
    
def visualize(image, points):
    imx = cv2.merge((image, image, image))
    for x, y in points:
        x = int(round(x))
        y = int(round(y))
        cv2.circle(imx, (x, y), 3, (255, 0, 0), thickness=1)
    return imx