import os
from os.path import join as pjoin
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import re



#读取文件夹中的数据，如读取train文件夹中的数据
def load_data_(data_path, class_list, size = 128, process_mod='diff', resize_mod='padding', augmentate=True):
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
                img_trgt = cv2.imread(pjoin(data_path, type, 'trgt', trgt_imgs[i]))
                img_temp = cv2.imread(pjoin(data_path, type, 'temp', temp_imgs[i]))
                #数据预处理

                # plt.imshow(img_trgt)
                # plt.show()
                # input('...')

                x = process(img_trgt, img_temp, process_mod)
                x = resize(x, size, resize_mod)

                # plt.imshow(x)
                # plt.show()
                # input('...')
                X.append(x)
                Y.append(y)
    return np.array(X), np.array(Y)

#读取train和test文件夹中的数据
def load_data(data_path=r'./data/cutted_data',size = 64, class_list = '[[1], [2], [14]]', process_mod='diff', resize_mod='padding', augmentate = True):
    class_list_ = eval(class_list)
    train_X, train_Y = load_data_(pjoin(data_path, 'train'), class_list_, size=size, process_mod=process_mod, resize_mod=resize_mod, augmentate = augmentate)
    test_X, test_Y = load_data_(pjoin(data_path, 'test'), class_list_, size=size, process_mod=process_mod, resize_mod=resize_mod, augmentate = augmentate)
    return train_X, train_Y, test_X, test_Y


def diff(img1, img2):
    return cv2.absdiff(img1,img2)


def resize(img0, size, mod = 'scale'):
    if mod == 'scale':
        img = cv2.resize(img0, dsize = (size,size))
    elif mod == 'padding':
        img_h = img0.shape[0]
        img_w = img0.shape[1]
        img_len = max(img_h, img_w)
        if img_len > size:
            f = size/img_len
            img0 = cv2.resize(img0,(0,0), fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
            img_h = img0.shape[0]
            img_w = img0.shape[1]
        x0 = int((size - img_w)/2)
        y0 = int((size - img_h)/2)
        img = np.zeros((size,size,3))
        img = np.array(img, dtype='uint8')
        img[y0:y0+img_h, x0:x0+img_w,:] = np.array(img0)


    return img

def process(img1,img2, mod):
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
        img = np.array(img, dtype='uint8')
        img = np.abs(img)


    elif mod == 'diff':
        img = diff(img1, img2)

    elif mod == 'trgt':
        img = img1
    return np.array(img)

if __name__ == '__main__':
    class_list = '[[4],[2],[14]]'
    # class_list = '[[1], [2], [3], [4], [5], [6], [7], [14], [16], [17], [20], [21], [22], [23], [24]]'
    class_num = len(eval(class_list))
    img_size = 400
    train_X, train_Y, test_X, test_Y = load_data(data_path=r'./data/cutted_data', size=256, process_mod='diff', resize_mod='padding',
                                                 class_list=class_list, augmentate=False)
    np.save('x_train', train_X)
    np.save('y_train', train_Y)
    np.save('x_test',test_X)
    np.save('y_test', test_Y)