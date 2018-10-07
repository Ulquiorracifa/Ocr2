import os
import sys
import importlib
import numpy as np
import cv2
import matplotlib.pyplot as pl
# %matplotlib inline

import fp.frame
import fp.train_ticket
import fp.core


def jwkj_get_filePath_fileName_fileExt(filename):  # 提取路径
    (filepath, tempfilename) = os.path.split(filename);
    (shotname, extension) = os.path.splitext(tempfilename);
    return filepath, shotname, extension


def detectType(filepath):
    rotateOn = True

    importlib.reload(fp)

    raw_im = cv2.imread(filepath, 1)
    # pl.figure(figsize=(12,12))
    # pl.imshow(raw_im)

    # 创建 发票票面 检测器 （无需再使用 Adjust 类）
    detect = fp.frame.surface.Detect(debug=False)  # 不要用debug模式
    # 检查是否上下颠倒火车票
    is_upside_down = fp.train_ticket.train_ticket.UpsideDownCheck_v2()

    if rotateOn:
        out_im = detect(raw_im)
        if out_im is None:
            print("Null output for {}".format(filepath))
            return
    else:
        out_im = cv2.imread(filepath, 1)
    # 转正
    # pl.figure(figsize=(12, 12))
    # pl.imshow(out_im, 'gray')

    importlib.reload(fp.train_ticket.train_ticket)
    # 检查是否蓝色票
    blue_ticket = fp.train_ticket.is_blue(out_im)
    # 是否上下颠倒，是则旋转180°
    # std_out_im = fp.train_ticket.train_ticket.turn_if_needed(out_im) # turn around if it's upside down
    # print("It is a {} ticket.".format('blue' if blue_ticket else 'red'))

    if rotateOn:
        if is_upside_down(out_im):
            # 旋转
            std_out_im = fp.core.trans.rotate180(out_im)
        else:
            std_out_im = out_im
    else:
        std_out_im = out_im

    if blue_ticket:
        flag = 1
        # print("type:  1 ")
    else:
        flag = 2
        # print("type:  2 ")
    # pl.figure(figsize=(12, 12))
    # pl.imshow(std_out_im, 'gray')

    '''if std_out_im.shape[0] > std_out_im.shape[1]:
        std_out_im = np.rot90(std_out_im)

    if rotateOn:
        adjust = fp.frame.surface.Adjust()
        std_out_im = adjust(std_out_im)

    print(std_out_im.shape[0])
    '''
    # 原
    # print(type(std_out_im))

    sfile = jwkj_get_filePath_fileName_fileExt(filepath)[0] + jwkj_get_filePath_fileName_fileExt(filepath)[
        1] + "_turned" + ".jpeg"
    cv2.imwrite(sfile, std_out_im)

    return sfile, flag


'''
dset_root = '/home/tangpeng/fapiao/dataset/scan_for_locate'
'''
# print(detectType('Image_00175.jpg'))

# test
# print(detectType('Image_065.jpg'))
