import os
from os.path import join as pjoin
import numpy as np
import matplotlib.pyplot as plt
import cv2



#读取文件夹中的数据，如读取train文件夹中的数据
def load_data_(data_path, class_list, size = 128):
    X = []
    Y = []
    train_dirs = os.listdir(data_path)
    for cls in range(len(class_list)):
        y = cls
        for t in class_list[cls]:
            type = 'type' + str(t)
            print('pre_processing: ', pjoin(data_path,type))
            trgt_imgs = os.listdir(pjoin(data_path, type, 'trgt'))
            temp_imgs = os.listdir(pjoin(data_path, type, 'temp'))
            for i in range(len(trgt_imgs)):
                # f_trgt, f_temp = open(pjoin(data_path, type, 'trgt', trgt_imgs[i]), 'rb'),  open(pjoin(data_path, type, 'temp', temp_imgs[i]), 'rb')
                img_trgt = cv2.imread(pjoin(data_path, type, 'trgt', trgt_imgs[i]))
                img_temp = cv2.imread(pjoin(data_path, type, 'temp', temp_imgs[i]))
                #数据预处理
                x = process(img_trgt, img_temp, size)
                X.append(x)
                Y.append(y)
                # f_trgt.close()
                # f_temp.close()
    return np.array(X), np.array(Y)

#读取train和test文件夹中的数据
def load_data(data_path=r'./data/cutted_data',size = 64, class_list = '[[1], [2], [14]]'):
    class_list_ = eval(class_list)
    train_X, train_Y = load_data_(pjoin(data_path, 'train'), class_list_, size=size)
    test_X, test_Y = load_data_(pjoin(data_path, 'test'), class_list_, size=size)
    return train_X, train_Y, test_X, test_Y


def diff(img1, img2):
    return cv2.absdiff(img1,img2)


def resize(img, size):
    img = cv2.resize(img, dsize = (size,size))
    return img

def process(img1,img2, size):
    img = diff(img1, img2)
    img = resize(img, size)
    return np.array(img)



