#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/12/6 2:50 下午
# @Author  : zbl
# @Email   : blzhu0823@gmail.com
# @File    : process.py
# @Software: PyCharm

import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from load_data import load_data
from os.path import join as pjoin


def read_img(img):
    np_img = np.array(img)
    return np_img


def cut(img, bbox):
    x0 = bbox['x0']
    y0 = bbox['y0']
    x1 = bbox['x1']
    y1 = bbox['y1']
    return img.crop((x0, y0, x1, y1))


def resize(img, size):
    return img.resize((size, size))


def diff(img1, img2):
    np_img1 = read_img(img1)
    np_img2 = read_img(img2)
    return np_img2 - np_img1


def process(img1, img2, bbox, size):
    img1 = cut(img1, bbox)
    img2 = cut(img2, bbox)
    diff_np_img = diff(resize(img1, size), resize(img2, size))
    return diff_np_img

def split_data(X, Y):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', default='./fabric_data', type=str)
    parser.add_argument('-saved_data_path', default='./processed_data', type=str)
    parser.add_argument('-size', default=400, type=int)
    parser.add_argument('-test_ratio', default=None, type=float)
    args = parser.parse_args()
    datas = load_data(args)
    task3_selected = [1, 2, 5, 13]
    X_task1 = []
    X_task2 = []
    X_task3 = []
    Y_task1 = []
    Y_task2 = []
    Y_task3 = []
    for data in datas:
        x = process(data['img1'], data['img2'], data['info']['bbox'], args.size)
        y = data['info']['flaw_type']
        X_task1.append(np.expand_dims(x, axis=0))
        X_task2.append(np.expand_dims(x, axis=0))
        Y_task1.append(y)
        if y >= 6 and y <= 12:
            Y_task2.append(6)
        elif y >= 9:
            Y_task2.append(y-6)
        else:
            Y_task2.append(y)
        if y in task3_selected:
            X_task3.append(np.expand_dims(x, axis=0))
            Y_task3.append(task3_selected.index(y))

    X_task1 = np.vstack(X_task1)
    X_task2 = np.vstack(X_task2)
    X_task3 = np.vstack(X_task3)
    Y_task1 = np.vstack(Y_task1).squeeze()
    Y_task2 = np.vstack(Y_task2).squeeze()
    Y_task3 = np.vstack(Y_task3).squeeze()

    if args.test_ratio is None:
        X_tasks = [X_task1, X_task2, X_task3]
        Y_tasks = [Y_task1, Y_task2, Y_task3]
        for i in range(3):
            if not os.path.exists(pjoin(args.saved_data_path, 'task'+str(i+1))):
                os.mkdir(pjoin(args.saved_data_path, 'task'+str(i+1)))
            np.save(pjoin(args.saved_data_path, 'task' + str(i+1), 'X'), X_tasks[i])
            np.save(pjoin(args.saved_data_path, 'task' + str(i+1), 'Y'), Y_tasks[i])
            print('X_train size for task' + str(i+1) + ':', X_tasks[i].shape)
            print('Y_train size for task' + str(i+1) + ':', Y_tasks[i].shape)

    else:
        X_tasks = [X_task1, X_task2, X_task3]
        Y_tasks = [Y_task1, Y_task2, Y_task3]
        for i in range(3):
            if not os.path.exists(pjoin(args.saved_data_path, 'task'+str(i+1))):
                os.mkdir(pjoin(args.saved_data_path, 'task'+str(i+1)))
            X_train, X_test, Y_train, Y_test = train_test_split(X_tasks[i], Y_tasks[i],
                                                                test_size=args.test_ratio,
                                                                random_state=42,
                                                                shuffle=True)
            np.save(pjoin(args.saved_data_path, 'task' + str(i+1), 'X_train'), X_train)
            np.save(pjoin(args.saved_data_path, 'task' + str(i + 1), 'X_test'), X_test)
            np.save(pjoin(args.saved_data_path, 'task' + str(i+1), 'Y_train'), Y_train)
            np.save(pjoin(args.saved_data_path, 'task' + str(i + 1), 'Y_test'), Y_test)
            print('X_train size for task' + str(i+1) + ':', X_train.shape)
            print('X_test size for task' + str(i + 1) + ':', X_test.shape)
            print('Y_train size for task' + str(i+1) + ':', Y_train.shape)
            print('Y_test size for task' + str(i + 1) + ':', Y_test.shape)




