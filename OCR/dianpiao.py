from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape, Masking, Lambda, Permute
from keras.layers import Input, Dense, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.layers.recurrent import GRU, LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Model
# from keras import backend as K
from keras.preprocessing import image
from keras.optimizers import Adam, SGD, Adadelta
from keras import losses
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.utils import plot_model
from matplotlib import pyplot as plt

import numpy as np
import os
from PIL import Image
import json
import threading

import tensorflow as tf
import keras.backend.tensorflow_backend as K

import cv2

import time

from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Reshape, Permute
from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Input, Flatten
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.wrappers import TimeDistributed

char = ''
with open(r'/home/huangzheng/ocr/OCR/rcnn_dic.txt', encoding='utf-8') as f:
    for ch in f.readlines():
        ch = ch.strip('\r\n')
        char = char + ch
char = char + '卍'
print('nclass:', len(char))
n_classes = len(char)

char_to_id = {j: i for i, j in enumerate(char)}
id_to_char = {i: j for i, j in enumerate(char)}

max_label_length = 12
img_h = 32
# img_w = 512
nclass = len(char)
rnnunit = 256
batch_size = 128


class Timer(object):
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def conv_block(input, growth_rate, dropout_rate=None, weight_decay=1e-4):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    x = Conv2D(growth_rate, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    if (dropout_rate):
        x = Dropout(dropout_rate)(x)
    return x


def dense_block(x, nb_layers, nb_filter, growth_rate, droput_rate=0.2, weight_decay=1e-4):
    for i in range(nb_layers):
        cb = conv_block(x, growth_rate, droput_rate, weight_decay)
        x = concatenate([x, cb], axis=-1)
        nb_filter += growth_rate
    return x, nb_filter


def transition_block(input, nb_filter, dropout_rate=None, pooltype=1, weight_decay=1e-4):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)

    if (dropout_rate):
        x = Dropout(dropout_rate)(x)

    if (pooltype == 2):
        x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    elif (pooltype == 1):
        x = ZeroPadding2D(padding=(0, 1))(x)
        x = AveragePooling2D((2, 2), strides=(2, 1))(x)
    elif (pooltype == 3):
        x = AveragePooling2D((2, 2), strides=(2, 1))(x)
    return x, nb_filter


def dense_cnn(input, nclass):
    _dropout_rate = 0.2
    _weight_decay = 1e-4

    _nb_filter = 64
    # conv 64  5*5 s=2
    x = Conv2D(_nb_filter, (5, 5), strides=(2, 2), kernel_initializer='he_normal', padding='same',
               use_bias=False, kernel_regularizer=l2(_weight_decay))(input)

    # 64 +  8 * 8 = 128
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
    # 128
    x, _nb_filter = transition_block(x, 128, _dropout_rate, 2, _weight_decay)

    # 128 + 8 * 8 = 192
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
    # 192->128
    x, _nb_filter = transition_block(x, 128, _dropout_rate, 2, _weight_decay)

    # 128 + 8 * 8 = 192
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)

    x, _nb_filter = transition_block(x, 128, _dropout_rate, 2, _weight_decay)

    # 128 + 8 * 8 = 192
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)

    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)

    x = Permute((2, 1, 3), name='permute')(x)
    x = TimeDistributed(Flatten(), name='flatten')(x)
    x = Bidirectional(GRU(rnnunit, return_sequences=True, implementation=2), name='blstm1')(x)
    # m = Bidirectional(LSTM(rnnunit,return_sequences=True),name='blstm1')(m)
    x = Dense(rnnunit, name='blstm1_out', activation='linear', )(x)
    # m = Bidirectional(LSTM(rnnunit,return_sequences=True),name='blstm2')(m)
    x = Bidirectional(GRU(rnnunit, return_sequences=True, implementation=2), name='blstm2')(x)
    y_pred = Dense(nclass, name='out', activation='softmax')(x)

    # basemodel = Model(inputs=input,outputs=y_pred)
    # basemodel.summary()
    return y_pred


def load_model():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    init = tf.global_variables_initializer()
    sess.run(init)

    input = Input(shape=(32, 512, 1), name='the_input')
    modelPath = r'/home/huangzheng/ocr/OCR/model/dianpiao_model.hdf5'
    print(modelPath)
    y_pred = dense_cnn(input, 17)
    basemodel = Model(inputs=input, outputs=y_pred)
    basemodel.load_weights(modelPath)
    return basemodel


def predict(img_path, base_model, thresholding=160):
    if thresholding > 255:
        thresholding = 255
    if thresholding < 0:
        thresholding = 0

    t = Timer()
    img = Image.open(img_path).convert('L')
    w, h = img.size
    rate = w / h

    img = img.resize((int(rate * 32), 32), Image.ANTIALIAS)

    img = np.array(img)
    # if thresholding == 0:
    #    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 5)
    # for i in range(32):
    #    for j in range(int(rate * 32)):
    #        if img[i,j] > 160:
    #            img[i,j] = 255
    #        else:
    #            img[i,j] = 0
    img = np.array(img, 'f') / 255.0 - 0.5
    t_img = np.zeros((32, 512))
    t_img[:, :int(rate * 32)] = img

    X = np.array([t_img])
    X = X.reshape((1, 32, 512, 1))
    t.tic()
    y_pred = base_model.predict(X)
    t.toc()
    print("times,", t.diff)
    argmax = np.argmax(y_pred, axis=2)[0]
    y_pred = y_pred[:, :, :]
    out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], )[0][0])[:, :]
    out = u''.join([id_to_char[x] for x in out[0]])

    return out, t_img


def OCR(image_path, base_model, color=1, thresholding=160):
    # base_model=load_model()
    """
        imgae_path 输入图片路径，识别图片为行提取结果
        color: 0 二值， 1 灰度， 2 彩色
        base_model 为加载模型，这个模型最好在服务器启动时加载，计算时作为参数输入即可，减少加载模型所需要的时间
    """
    out, _ = predict(image_path, base_model, thresholding=thresholding)

    return out


if __name__ == "__main__":
    file_path = "./Val/"
    i = 1
    base_model = load_model()
    images_file = os.listdir(file_path)
    for img in images_file[:10]:
        image = file_path + img
        print(img)
        out = OCR(image, base_model)
        print(out)
