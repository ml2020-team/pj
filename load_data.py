import os
from os.path import join as pjoin
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import cv2
import re



#读取文件夹中的数据，如读取train文件夹中的数据
def load_data_(data_path, class_list, size = 128, mod='diff', augmentate=True):
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
                if augmentate == False:
                    if re.match('aug', trgt_imgs[i]):
                        continue
                # f_trgt, f_temp = open(pjoin(data_path, type, 'trgt', trgt_imgs[i]), 'rb'),  open(pjoin(data_path, type, 'temp', temp_imgs[i]), 'rb')
                img_trgt = cv2.imread(pjoin(data_path, type, 'trgt', trgt_imgs[i]))
                img_temp = cv2.imread(pjoin(data_path, type, 'temp', temp_imgs[i]))
                #数据预处理
                x = process(img_trgt, img_temp, size, mod)
                X.append(x)
                Y.append(y)
                # f_trgt.close()
                # f_temp.close()
    return np.array(X), np.array(Y)

#读取train和test文件夹中的数据
def load_data(data_path=r'./data/cutted_data',size = 64, class_list = '[[1], [2], [14]]', mod='diff', augmentate = True):
    class_list_ = eval(class_list)
    train_X, train_Y = load_data_(pjoin(data_path, 'train'), class_list_, size=size, mod=mod, augmentate = augmentate)
    test_X, test_Y = load_data_(pjoin(data_path, 'test'), class_list_, size=size, mod=mod, augmentate = augmentate)
    return train_X, train_Y, test_X, test_Y


def diff(img1, img2):
    return cv2.absdiff(img1,img2)


def resize(img, size):
    img = cv2.resize(img, dsize = (size,size))
    return img

def process(img1,img2, size, mod):
    shape = np.array(img1).shape
    if mod == 'avg_pool' or mod == 'max_pool':
        img1, img2 =\
            torch.from_numpy(np.array(img1)),\
            torch.from_numpy(np.array(img2))
        img1_r, img1_g, img1_b = img1[:,:,0], img1[:,:,1], img1[:,:,2]
        img2_r, img2_g, img2_b = img2[:,:,0], img2[:,:,1], img2[:,:,2]
        img1_c = [img1_r, img1_g, img1_b]
        img2_c = [img2_r, img2_g, img2_b]
        img_c_ = img1_c + img2_c
        img_c = []

        for i in img_c_:
            if mod == 'max_pool':
                i = F.max_pool2d(i.unsqueeze(0).float(), kernel_size=3, stride=1).numpy()
            elif mod == 'avg_pool':
                i  = F.avg_pool2d(i.unsqueeze(0).float(), kernel_size=3, stride=1).numpy()
            i = i[0,:,:]
            i  = i[:,:,np.newaxis]
            img_c.append(i)


        img1 = np.concatenate((img_c[0], img_c[1], img_c[2]), axis=2)
        img2 = np.concatenate((img_c[3], img_c[4], img_c[5]), axis=2)
        img = img1 - img2
        # img = np.abs(img)
        img = resize(img, size)


    elif mod == 'diff':
        img = diff(img1, img2)
        img = resize(img, size)
    return np.array(img)

if __name__ == '__main__':
    class_list = '[[1],[2],[14]]'
    # class_list = '[[1], [2], [3], [4], [5], [6], [7], [14], [16], [17], [20], [21], [22], [23], [24]]'
    class_num = len(eval(class_list))
    img_size = 50
    train_X, train_Y, test_X, test_Y = load_data(data_path=r'./data/cutted_data', size=img_size, mod='max_pool',
                                                 class_list=class_list, augmentate=True)
    np.save('x_train', train_X)
    np.save('y_train', train_Y)
    np.save("x_test",test_X)
    np.save('y_test', test_Y)