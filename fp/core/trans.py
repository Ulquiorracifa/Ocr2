import numpy as np
import cv2


def rotate_crop(src_image, center, angle, dsize, border_color=(255,255,255)):
    trans_mat = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    trans_mat[0, 2] -= (center[0] - 0.5 * dsize[0])
    trans_mat[1, 2] -= (center[1] - 0.5 * dsize[1])
    img = cv2.warpAffine(src_image, trans_mat, dsize, borderValue=border_color)
    return img

def box_crop(src_image, box, border_color=(255,255,255)):
    center, dsize, angle = box
    trans_mat = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    trans_mat[0, 2] -= (center[0] - 0.5 * dsize[0])
    trans_mat[1, 2] -= (center[1] - 0.5 * dsize[1])
    dsize = int(dsize[0]), int(dsize[1])
    img = cv2.warpAffine(src_image, trans_mat, dsize, borderValue=border_color)
    return img

# delete this if no one use this
# use 'deskew' instead
def warp(src_image, src_points, dsize, border_color=(255,255,255)):
    w, h = 1. * dsize
    src_points = np.array(src_points).astype(np.float32)
    dst_points = np.array([[0,0], [0,h], [w,h], [w,0]]).astype(np.float32)
    trans_mat = cv2.getPerspectiveTransform(src_points, dst_points)
    img = cv2.warpPerspective(src_image, trans_mat, dsize, borderValue=border_color)
    return img

def deskew(image, box_points, dsize=None, border_color=(255,255,255)):
    # 'box_point' must be in a sequence-order of
    #      1  2
    #      0  3
    box_points = np.array(box_points, dtype=np.float32)
    if dsize is None:
        p0, p1, p2, p3 = box_points
        d0 = max(np.linalg.norm(p0 - p3), np.linalg.norm(p1 - p2)) # width
        d1 = max(np.linalg.norm(p0 - p1), np.linalg.norm(p3 - p2)) # height 
        dsize = int(round(d0)), int(round(d1))
    else:
        d0, d1 = dsize
    dst_points = np.array([[0, d1-1], [0, 0], [d0-1, 0], [d0-1, d1-1]], 
                          dtype=np.float32)
    
    # calculate the perspective transform matrix and warp
    M = cv2.getPerspectiveTransform(box_points, dst_points)
    deskew_im = cv2.warpPerspective(image, M, dsize, borderValue=border_color)
    return deskew_im
        
def rotate180(image):
    '''rotate the image in 180 degrees, used when the image is upside down'''
    return np.fliplr(np.flipud(image))

def rotate90(image):
    return np.rot90(image, k=1, axes=(0, 1))

def shift(image, tlx, tly):
    '''if top-left x(y) is negative, shift to bottom-right
    '''
    h, w = image.shape[:2]
    sx0 = max(tlx, 0)
    sx1 = min(tlx + w, w)
    sy0 = max(tly, 0)
    sy1 = min(tly + h, h)
    dx0 = sx0 - tlx
    dx1 = sx1 - tlx
    dy0 = sy0 - tly
    dy1 = sy1 - tly
    imagex = image.copy()
    imagex[dy0:dy1, dx0:dx1] = image[sy0:sy1, sx0:sx1]
    return imagex

def pix_scale(image, pix_min=0, pix_max=255):
    im_min, im_max = np.min(image), np.max(image)
    image = (image.astype(np.float32) - im_min) / (im_max - im_min)
    image = (pix_max - pix_min) * image + pix_min
    image = np.round(image).astype(np.uint8)
    return image