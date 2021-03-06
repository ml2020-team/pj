import math
import os
import argparse
import pickle
import random

import numpy as np
from sklearn.model_selection import train_test_split
from load_raw_data import load_data 
from augmentation import transform_image
from os.path import join as pjoin
from PIL import Image
import shutil
import tensorflow as tf


def read_img(img):
    np_img = np.array(img)
    return np_img


def cut(img, bbox):
    x0 = bbox['x0']
    y0 = bbox['y0']
    x1 = bbox['x1']
    y1 = bbox['y1']
    return img.crop((x0, y0, x1, y1))


def process(img1, img2, bbox):
    img1 = cut(img1, bbox)
    img2 = cut(img2, bbox)
    return img1, img2


def catagory(flaw_type, flaw_count, test_ratio = 0.2):
    if os.path.exists(args.saved_data_path):
        shutil.rmtree(args.saved_data_path)
        print(args.saved_data_path)
    if not os.path.exists(args.saved_data_path):
        os.mkdir(args.saved_data_path)

    if not os.path.exists(pjoin(args.saved_data_path, 'test')):
        os.mkdir(pjoin(args.saved_data_path, 'test'))
    if not os.path.exists(pjoin(args.saved_data_path, 'train')):
            os.mkdir(pjoin(args.saved_data_path, 'train'))

    for i in range(15):
            os.mkdir(pjoin(args.saved_data_path, 'test', 'type' + str(flaw_type[i])))
            os.mkdir(pjoin(args.saved_data_path, 'train', 'type' + str(flaw_type[i])))
            os.mkdir(pjoin(args.saved_data_path, 'test', 'type' + str(flaw_type[i]), 'temp'))
            os.mkdir(pjoin(args.saved_data_path, 'test', 'type' + str(flaw_type[i]), 'trgt'))
            os.mkdir(pjoin(args.saved_data_path, 'train', 'type' + str(flaw_type[i]), 'temp'))
            os.mkdir(pjoin(args.saved_data_path, 'train', 'type' + str(flaw_type[i]), 'trgt'))
    print('loading_datas')
    datas = load_data(args)
    random.shuffle(datas)
    for data in datas:
        y = data['info']['flaw_type']
        flaw_count[flaw_type.index(y)] += 1

    print(flaw_count)
    flaw_count_tmp =[0]*len(flaw_count)
    for data in datas:
        y = data['info']['flaw_type']

        img1, img2 = process(data['img1'], data['img2'], data['info']['bbox'])
        if flaw_count_tmp[flaw_type.index(y)] < int(flaw_count[flaw_type.index(y)] * test_ratio):
            print('save','type' + str(y), 'pic' + str(flaw_count_tmp[flaw_type.index(y)]) + '.jpg')
            img1.save(
                pjoin(args.saved_data_path, 'test', 'type' + str(y), 'temp', 'pic' + str(flaw_count_tmp[flaw_type.index(y)]) + '.jpg'))
            img2.save(
                pjoin(args.saved_data_path, 'test', 'type' + str(y), 'trgt', 'pic' + str(flaw_count_tmp[flaw_type.index(y)]) + '.jpg'))
        else:
            print('save','type' + str(y), 'pic' + str(flaw_count_tmp[flaw_type.index(y)]) + '.jpg')
            img1.save(
                pjoin(args.saved_data_path, 'train', 'type' + str(y), 'temp', 'pic' + str(flaw_count_tmp[flaw_type.index(y)]) + '.jpg'))
            img2.save(
                pjoin(args.saved_data_path, 'train', 'type' + str(y), 'trgt', 'pic' + str(flaw_count_tmp[flaw_type.index(y)]) + '.jpg'))

        flaw_count_tmp[flaw_type.index(y)] += 1


def aug_collection(flaw_type, flaw_count, test_ratio=0.2):
    # collection1 = [1, 2, 4, 14, 20]
    # collection1 = [14]
    print("aug_collection")
    session = tf.Session()
    print("tf.session")
    for type in flaw_type:
        flag = False
        times = math.ceil(600.0 / (flaw_count[flaw_type.index(type)]*(1-test_ratio) - 1) - 1)
        PATH = pjoin(args.saved_data_path,'train', 'type' + str(type))
        paths = os.listdir(pjoin(PATH, 'temp'))
        for path in paths:
            print("aug_collection: ", path)
            f1 = open(pjoin(PATH, 'temp', path),'rb')
            f2 = open(pjoin(PATH, 'trgt', path),'rb')
            img1 = Image.open(f1)
            img2 = Image.open(f2)
            img1.load()
            img2.load()
            new_image_count = 0
            for time in range(times):
                print('type',type,' ', path, "生成新图片...", new_image_count)
                new_image_count += 1
                img_array1 = np.array(img1)
                img_array2 = np.array(img2)
                transformed_img1, transformed_img2 = transform_image(img_array1, img_array2)
                Image.fromarray(transformed_img1.eval(session=session)).save(
                    pjoin(PATH, 'temp', 'augpic' + str(flaw_count[flaw_type.index(type)]) + '.jpg'))
                Image.fromarray(transformed_img2.eval(session=session)).save(
                    pjoin(PATH, 'trgt', 'augpic' + str(flaw_count[flaw_type.index(type)]) + '.jpg'))
                if flaw_count[flaw_type.index(type)] == 600:
                    flag = True
                    break
                flaw_count[flaw_type.index(type)] += 1
            f1.close()
            f2.close()
            if flag:
                break
    session.close()


def split_data(flaw_type, file_path):
    print("split_data")
    train_data, test_data = [], []
    for type in flaw_type:
        paths = os.listdir(pjoin(args.saved_data_path, 'type' + str(type), 'temp'))
        data = [[type, paths[i]] for i in range(len(paths))]
        train, test = train_test_split(data, test_size=args.test_ratio, random_state=42, shuffle=True)
        train_data += train
        test_data += test

    random.shuffle(train_data)
    random.shuffle(test_data)

    with open(pjoin(file_path, "train_data.txt"), "wb+") as f:
        pickle.dump(train_data, f)
    with open(pjoin(file_path, "test_data.txt"), "wb+") as f:
        pickle.dump(test_data, f)


def conv2numpy(file_path, flaw_type):
    with open(pjoin(file_path, "train_data.txt"), "rb") as f:
        train_data = pickle.load(f)
    with open(pjoin(file_path, "test_data.txt"), "rb") as f:
        test_data = pickle.load(f)

    x_train1, y_train, x_test1, y_test = [], [], [], []
    x_train2, x_test2 = [], []
    for i in train_data:
        print("conv2numpy_train: ", pjoin(file_path, 'type' + str(i[0])))
        img1 = Image.open(pjoin(file_path, 'type' + str(i[0]), 'temp', i[1]))
        x_train1.append(np.expand_dims(np.array(img1), axis=0))
        y_train.append(flaw_type.index(i[0]))
        img2 = Image.open(pjoin(file_path, 'type' + str(i[0]), 'trgt', i[1]))
        x_train2.append(np.expand_dims(np.array(img2), axis=0))
    for i in test_data:
        print("conv2numpy_test: ", pjoin(file_path, 'type' + str(i[0])))
        img1 = Image.open(pjoin(file_path, 'type' + str(i[0]), 'temp', i[1]))
        x_test1.append(np.expand_dims(np.array(img1), axis=0))
        y_test.append(flaw_type.index(i[0]))
        img2 = Image.open(pjoin(file_path, 'type' + str(i[0]), 'trgt', i[1]))
        x_test2.append(np.expand_dims(np.array(img2), axis=0))

    x_train1 = np.vstack(x_train1)
    x_test1 = np.vstack(x_test1)
    x_train2 = np.vstack(x_train2)
    x_test2 = np.vstack(x_test2)
    y_train = np.vstack(y_train).squeeze()
    y_test = np.vstack(y_test).squeeze()

    np.save(pjoin(file_path, 'x_train1'), x_train1)
    np.save(pjoin(file_path, 'x_test1'), x_test1)
    np.save(pjoin(file_path, 'x_train2'), x_train2)
    np.save(pjoin(file_path, 'x_test2'), x_test2)
    np.save(pjoin(file_path, 'y_train'), y_train)
    np.save(pjoin(file_path, 'y_test'), y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', default='./data/fabric_data_new', type=str)
    parser.add_argument('-saved_data_path', default='./data/cutted_data', type=str)
    parser.add_argument('-size', default=50, type=int)
    parser.add_argument('-test_ratio', default=0.2, type=float)
    args = parser.parse_args()
    flaw_type = [1, 2, 3, 4, 5, 6, 7, 14, 16, 17, 20, 21, 22, 23, 24]
    flaw_count = [1 for i in range(15)]

    catagory(flaw_type, flaw_count)

    aug_collection(flaw_type, flaw_count)
    #
    #split_data(flaw_type, "./data")
    #
    #conv2numpy(flaw_type, "./data")
