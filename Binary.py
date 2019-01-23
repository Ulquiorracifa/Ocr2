import fp
import cv2
import os

if __name__ == '__main__':
    dest = '/home/huangzheng/labeledimage/'
    resdest = '/home/huangzheng/labeledimage/tmp/'
    jpgs = fp.util.path.files_in_dir(dest, '.jpg')
    for c in jpgs:
        imcv = cv2.imread(os.path.join(dest, c), 1)[:, :, 2]
        threshold = fp.core.thresh.HybridThreshold(rows=1, cols=4, local='gauss')
        bi_im = threshold(imcv)
        cv2.imwrite(os.path.join(resdest, c), bi_im)
        print(c + 'done.  ' + os.path.join(resdest, c))
