from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Permute
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.layers import Input, Flatten
from keras.regularizers import l2
import keras.backend.tensorflow_backend as K
import os

import numpy as np
import tensorflow as tf
from PIL import Image

dic_path = '/home/luoyc/ocr/OCR/rcnn_dic.txt'
char = ''
with open(dic_path, encoding='utf-8') as f:
    for ch in f.readlines():
        ch = ch.strip('\r\n')
        char = char + ch
char = char + '卍'
print('nclass:', len(char))
n_classes = len(char)
char_to_id = {j: i for i, j in enumerate(char)}
id_to_char = {i: j for i, j in enumerate(char)}


def conv_block(x, filters, dropout_rate=None):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x


def dense_block(x, layer_count, filters, growth_num, dropout_rate=0.2):
    for i in range(layer_count):
        cb = conv_block(x, growth_num, dropout_rate)
        x = concatenate([x, cb], axis=-1)
        filters += growth_num
    return x, filters


def transition_block(x, filters, dropout_rate=0.2, pooltype=1, weight_decay=1e-4):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    if pooltype == 1:
        x = ZeroPadding2D(padding=(0, 1))(x)
        x = MaxPooling2D((2, 2), strides=(2, 1))(x)
    elif pooltype == 2:
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    elif pooltype == 3:
        x = MaxPooling2D((2, 2), strides=(2, 1), padding='same')(x)
    return x, filters


def dense_cnn(input, n_classes):
    _dropout_rate = 0.1
    _weight_decay = 1e-4
    _first_filters = 64
    # input 32 * W * 1
    x = Conv2D(_first_filters, (3, 3), strides=(1, 1), kernel_initializer='he_normal',
               padding='same', use_bias=False, kernel_regularizer=l2(_weight_decay))(input)  # 32 * W * 64
    x = MaxPooling2D((2, 2))(x)  # 16 * (W/2) * 64

    x, filters = dense_block(x, 8, _first_filters, 8, _dropout_rate)  # 16 * (W/2) * 128
    x, filters = transition_block(x, filters, _dropout_rate, 3, _weight_decay)  # 8 * (W/2) * 128

    x, filters = dense_block(x, 8, filters, 8, _dropout_rate)  # 8 * (W/2) * 196
    x, filters = transition_block(x, filters, _dropout_rate, 3, _weight_decay)  # 4 * (W/2) * 196

    x, filters = dense_block(x, 8, filters, 8, _dropout_rate)  # 4 * (W/2) * 256
    x, filters = transition_block(x, filters, _dropout_rate, 2, _weight_decay)  # 2 * (W/4) * 256

    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)

    x = Permute((2, 1, 3), name='permute')(x)
    x = TimeDistributed(Flatten(), name='flatten')(x)
    y_pred = Dense(n_classes, name='out', activation='softmax')(x)
    return y_pred


def load_model(n_classes=17, model_path='/home/luoyc/ocr/OCR/model/ocr3_model.hdf5', max_label_length=25,
               input_shape=(32, None, 1)):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto(allow_soft_placement=True)
    # # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    #
    # # init = tf.global_variables_initializer()
    # # sess.run(init)
    K.set_session(sess)
    input = Input(input_shape, name="input")
    print('load model: {}'.format(model_path))
    y_pred = dense_cnn(input, 17)
    basemodel = Model(inputs=input, outputs=y_pred)
    basemodel.load_weights(model_path)
    return basemodel


def predict(img_path, base_model):
    img = Image.open(img_path).convert('L')
    w, h = img.size
    rate = w / h

    img = img.resize((int(rate * 32), 32), Image.ANTIALIAS)
    img = np.array(img).astype(np.float32) / 255.0 - 0.5
    x = img.reshape(1, 32, int(rate * 32), 1)
    y_pred = base_model.predict(x)
    print(np.argmax(y_pred, axis=2)[0])
    y_pred = y_pred[:, :, :]
    print(type(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], )[0][0]))
    out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], )[0][0])[:, :]
    out = u''.join([id_to_char[x] for x in out[0]])
    return out, img


def OCR(image_path, base_model, color=1, thresholding=160):
    # base_model=load_model()
    """
        imgae_path 输入图片路径，识别图片为行提取结果
        color: 0 二值， 1 灰度， 2 彩色
        base_model 为加载模型，这个模型最好在服务器启动时加载，计算时作为参数输入即可，减少加载模型所需要的时间
    """
    out, _ = predict(image_path, base_model)

    return out


if __name__ == "__main__":
    file_path = "./valid/"
    i = 1
    base_model = load_model()
    images_file = os.listdir(file_path)
    for img in images_file[:10]:
        image = file_path + img
        print(img)
        out = OCR(image, base_model)
        print(out)
