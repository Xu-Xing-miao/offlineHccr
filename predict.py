"""
inference on a single Chinese character
image and recognition the meaning of it
"""
import os
import cv2
import numpy as np
import tensorflow as tf
import glob
from alfred.dl.tf.common import mute_tf
mute_tf()
from alfred.utils.log import logger as logging
from dataset_prepared.casia_hwdb import load_characters
from models.architecture.cnn_net import build_net_004



target_size = 96
characters = load_characters()
num_classes = len(characters)
#use_keras_fit = False
use_keras_fit = True
ckpt_path = 'results/completed_model/1.1-epoch-25-lr-0.001/Melnyk-25.h5'


def show(name, src):
    """
        展示图片 show
        :param name:图片
        :param src:宽度
        :return:展示的图片
    """
    cv2.imshow(name, src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):  #
    """
    使用插值方法对图片 resize
    :param image:图片
    :param width:宽度
    :param height:高度
    :param inter:插值方法
    :return:调整大小后的图片
    """
    dim = None
    (h, w) = image.shape[:2]  #
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    # print(dim)
    resized = cv2.resize(image, dim, interpolation=inter)  # interpo;ation为插值方法，这里选用的是
    return resized


def preprocess(x):
    """
    minus mean pixel or normalize?
    """
    # original is 64x64, add a channel dim
    x['image'] = tf.expand_dims(x['image'], axis=-1)
    x['image'] = tf.image.resize(x['image'], (target_size, target_size))
    x['image'] = (x['image'] - 128.) / 128.
    return x['image'], x['label']


def get_model():
    # init model
    model = build_net_004((96, 96, 1), num_classes)
    logging.info('model loaded.')

    latest_ckpt = tf.train.latest_checkpoint(os.path.dirname(ckpt_path))
    if latest_ckpt:
        #start_epoch = int(latest_ckpt.split('-')[1].split('.')[0])
        model.load_weights(latest_ckpt)
        logging.info('model resumed from: {}'.format(latest_ckpt))
        return model
    else:
        logging.error('can not found any models matched: {}'.format(ckpt_path))


def predict(model, img_f):
    ori_img = cv2.imread(img_f)
    # show("r", ori_img)

    img = tf.expand_dims(ori_img[:, :, 0], axis=-1)
    img = tf.image.resize(img, (target_size, target_size))
    img = (img - 128.)/128.
    img = tf.expand_dims(img, axis=0)
    print(img.shape)

    out = model(img).numpy()
    print('{}'.format(characters[np.argmax(out[0])]))
    name = '{}'.format(characters[np.argmax(out[0])])
    # 采用ASCll码代替无法创建文件的符号
    if name == '\\' or name == '/' or name == ':' or name == '*' or name == '?' \
            or name == '"' or name == '<' or name == '>' or name == '|':
        name = str(ord(name))

    pos1 = img_f.rfind('.')
    pos2 = img_f.rfind('\\')
    file_name = img_f[pos2 + 1:pos1]

    pre_name = file_name + "--" + name + ".jpg"

    # support data
    path = 'dataset/results/'
    if not os.path.exists(path):
        os.makedirs(path)
    cv2.imencode('.jpg', ori_img)[1].tofile(path + pre_name)


def get_file_name(filename):
    (filepath, tempFileName) = os.path.split(filename)
    (shotName, extension) = os.path.splitext(tempFileName)
    return shotName


if __name__ == '__main__':
    img_files = glob.glob('D:/pycharm/dataset/single_word/test/dataset-1/*.jpg')

    model = get_model()
    for img_f in img_files:
        a = cv2.imread(img_f)
        predict(model, img_f)




