''' 
The functions normalize_bitmap, tagcode_to_unicode and unicode_to_tagcode
are written by Alessandro and Francesco https://github.com/integeruser/CASIA-HWDB1.1-cnn
'''
import os
import struct
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import h5py
from scipy.io import loadmat
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from skimage.filters import threshold_otsu
from glob import glob
from tensorflow.keras.layers import (
	Input,
	Activation,
	Dense,
	Flatten,
	GlobalAveragePooling2D
	)


IMG_SHAPE = (96, 96)


# 全局加权平均池化
class GlobalWeightedAveragePooling2D(GlobalAveragePooling2D):# 继承GlobalAveragePooling2D函数
	def __init__(self, kernel_initializer='uniform', **kwargs):# def__init__()将需要绑定的属性强制填写进去。
		self.kernel_initializer = kernel_initializer
		super(GlobalWeightedAveragePooling2D, self).__init__(**kwargs)# 采用super引入另一种参数，**kwargs为一个字典类型(dict)

	def build(self, input_shape):# 定义权重
		# 为该层创建可训练的权重
		self.W = self.add_weight(name='W',
								 shape=input_shape[1:],
								 initializer=self.kernel_initializer,
								 trainable=True)
		# print('input_shape:', input_shape)
		super(GlobalWeightedAveragePooling2D, self).build(input_shape)# 最后调用super().build()

	def call(self, inputs):# 编写层的功能逻辑
		# 输入的每个项的按元素乘法
		inputs = inputs*self.W  # 全局平均池化后的特征图直接乘以权重参数
		if self.data_format == 'channels_last':
			return K.sum(inputs, axis=[1, 2])
		else:
			return K.sum(inputs, axis=[2, 3])

# 全局加权平均输出池化
class GlobalWeightedOutputAveragePooling2D(GlobalAveragePooling2D):

	def __init__(self, kernel_initializer='uniform', **kwargs):
		self.kernel_initializer = kernel_initializer
		super(GlobalWeightedOutputAveragePooling2D, self).__init__(**kwargs)

	def build(self, input_shape):
		# tensorflow的输入数据的格式((样本维数，高，宽，通道数))
		# 调用theano库输入数据的格式为((样本维数，通道数，高，宽))
		if self.data_format == 'channels_last':
			kernel_shape = [input_shape[-1]]# 卷积核大小跟通道数一样
		else:
			kernel_shape = [input_shape[1]]

		self.W = self.add_weight(name='W',
								 shape=kernel_shape,
								 initializer=self.kernel_initializer,
								 trainable=True)
		# print('input_shape:', input_shape).
		super(GlobalWeightedOutputAveragePooling2D, self).build(input_shape)

	def call(self, inputs):
		inputs = inputs*self.W # element-wise multiplication for every entry of input
		if self.data_format == 'channels_last':
			return K.sum(inputs, axis=[1, 2])
		else:
			return K.sum(inputs, axis=[2, 3])


def normalize_bitmap(bitmap):
	# pad the bitmap to make it squared 填充位图用以平方计算
	pad_size = abs(bitmap.shape[0]-bitmap.shape[1]) // 2
	if bitmap.shape[0] < bitmap.shape[1]:
		pad_dims = ((pad_size, pad_size), (0, 0))
	else:
		pad_dims = ((0, 0), (pad_size, pad_size))
	bitmap = np.lib.pad(bitmap, pad_dims, mode='constant', constant_values=255)

	# rescale and add empty border 重新缩放并添加空白边框
	bitmap = sp.misc.imresize(bitmap, (96 - 4 * 2, 96 - 4 * 2))
	bitmap = np.lib.pad(bitmap, ((4, 4), (4, 4)), mode='constant', constant_values=255)
	assert bitmap.shape == IMG_SHAPE

	bitmap = np.expand_dims(bitmap, axis=0)
	assert bitmap.shape == (1, *IMG_SHAPE)
	return bitmap


def preprocess_bitmap(bitmap):
	# inverse the gray values:
	bitmap = 255 - bitmap
	return bitmap


def tagcode_to_unicode(tagcode):
	return struct.pack('>H', tagcode).decode('gbk')


def unicode_to_tagcode(tagcode_unicode):
	return struct.unpack('>H', tagcode_unicode.encode('gbk'))[0]


def rgb2gray(rgb):
	return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

	
def preprocess_input(img, show_img=False):
		'''
		Preprocesses an input image for futrther prediction
		'''	
		if show_img:
			print('\nOriginal image shape:', img.shape)
			plt.imshow(img)
			plt.title('original')
			plt.show()

		gray_img = rgb2gray(img)
		thresh = threshold_otsu(gray_img)
		# binarize to get a 'white' (255) background:
		if np.mean(gray_img > thresh) > np.mean(gray_img < thresh):
			gray_img[gray_img > thresh] = 255
		else: 
			gray_img[gray_img < thresh] = 0
			gray_img = 255 - gray_img 

		norm_img = normalize_bitmap(np.array(gray_img, dtype=np.uint8))
		prepr_img = np.array(preprocess_bitmap(norm_img), dtype=np.uint8).reshape(1, *IMG_SHAPE, 1)
		if show_img:
			
			print('Preprocessed image shape:', prepr_img.shape)
			plt.imshow(prepr_img.reshape(*IMG_SHAPE), cmap='gray')
			plt.title('preprocessed')
			plt.show()

		return prepr_img


def load_data(image_path=None):
	print('\nLoading the data.....................................')
	if image_path and image_path[-4:].lower() in ['.png', '.jpg', 'jpeg']:
		img = np.array(image.load_img(image_path))
		prepr_img = preprocess_input(img, show_img=False)
		print()

		return [img], prepr_img
		
	elif image_path:
		files = glob(image_path + '/*.png') # for .png
		files += glob(image_path + '/*.jp*g') # append for .jpg and .jped
		n_files = len(files)
		print('Found %d images in the specified directory\n' % n_files)
		# print(files)
				
		prepr_imgs = np.empty((n_files, 96, 96, 1))
		imgs = []
		for i, img_filepath in enumerate(files):
			img = np.array(image.load_img(img_filepath))
			imgs.append(img)
			prepr_img = preprocess_input(img, show_img=False)
			prepr_imgs[i] = prepr_img	

		return imgs, prepr_imgs


def get_mnist_data(reshape=False, normalize=False):
	# load the MNIST data:
	(Xtrain, Ytrain), (Xtest, Ytest) = mnist.load_data()
	Xtrain = np.expand_dims(Xtrain, axis=3)
	Xtest = np.expand_dims(Xtest, axis=3)
	
	if reshape:
		N, K = len(Ytrain), len(set(Ytrain))
		D = Xtrain.shape[1]*Xtrain.shape[2]
		# reshape the data to be (NxD):
		Xtrain, Xtest = Xtrain.reshape(N, D), Xtest.reshape(len(Xtest), D)

	if normalize:
		Xtrain = np.float32(Xtrain / 255.0)
		Xtest = np.float32(Xtest / 255.0)

	return (Xtrain, Ytrain), (Xtest, Ytest)


def get_fashion_mnist_data(reshape=False, normalize=False, add_label_names=False):
	# load the Fashion-MNIST data:
	(Xtrain, Ytrain), (Xtest, Ytest) = fashion_mnist.load_data()	
	Xtrain = np.expand_dims(Xtrain, axis=3)
	Xtest = np.expand_dims(Xtest, axis=3)
	
	if reshape:
		N, K = len(Ytrain), len(set(Ytrain))
		D = Xtrain.shape[1]*Xtrain.shape[2]
		# reshape the data to be (NxD):
		Xtrain, Xtest = Xtrain.reshape(N, D), Xtest.reshape(len(Xtest), D)

	if normalize:
		Xtrain = np.float32(Xtrain / 255.0)
		Xtest = np.float32(Xtest / 255.0)

	if add_label_names:
		label_names = ['T-shirt/top', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
		return (Xtrain, Ytrain), (Xtest, Ytest), label_names

	return (Xtrain, Ytrain), (Xtest, Ytest)
