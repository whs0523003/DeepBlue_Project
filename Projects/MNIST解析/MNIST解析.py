import os
import struct
import sys

import numpy as np
import matplotlib.pyplot as plt


'''
TRAINING SET LABEL FILE (train-labels-idx1-ubyte):

[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
0004     32 bit integer  60000            number of items 
0008     unsigned byte   ??               label 
0009     unsigned byte   ??               label 
........ 
xxxx     unsigned byte   ??               label

The labels values are 0 to 9.
'''


# 解析图片
def load_image(path):
    # 加载训练集图片
    images_train_path = os.path.join(path, 'data/train-images.idx3-ubyte')
    # 加载测试集图片
    images_test_path = os.path.join(path, 'data/t10k-images.idx3-ubyte')

    # 读取训练集
    with open(images_train_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        '''
        np.tofile()将数组中的数据以二进制格式写进文件，输出的数据不保存数组形状和元素类型等信息
        np.fromfile()函数读回数据时需要用户指定元素类型，并对数组的形状进行适当的修改
        '''
        images_train = np.fromfile(imgpath, dtype=np.uint8).reshape(60000, 784)

    # 读取测试集
    with open(images_test_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images_test = np.fromfile(imgpath, dtype=np.uint8).reshape(10000, 784)

    return images_train, images_test


# 解析标签
def load_label(path):
    # 加载训练集标签
    labels_train_path = os.path.join(path, 'data/train-labels.idx1-ubyte')
    # 加载测试集标签
    labels_test_path = os.path.join(path, 'data/t10k-labels.idx1-ubyte')

    # 读取8位，即前2个元素。第1个magic代表幻数，即文件协议格式，第2个n代表数据集大小，为60000
    with open(labels_train_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels_train = np.fromfile(lbpath, dtype=np.uint8)

    with open(labels_test_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels_test = np.fromfile(lbpath, dtype=np.uint8)

    return labels_train, labels_test


# 画图
def plot_mnist():
    # 2行5列
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()

    # 拿出每个类别的第一张图片
    for i in range(10):
        img_train = images_train[labels_train == i][0].reshape(28, 28)
        img_test = images_test[labels_test == i][0].reshape(28, 28)
        ax[i].imshow(img_train, cmap='Greys', interpolation='nearest')
        ax[i].imshow(img_test, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])

    # tight_layout会自动调整子图参数，使之填充整个图像区域
    plt.tight_layout()
    plt.show()


# 主函数
if __name__ == '__main__':
    images_train, images_test = load_image('./')
    labels_train, labels_test = load_label('./')
    plot_mnist()