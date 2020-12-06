#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/12/6 2:50 下午
# @Author  : zbl
# @Email   : blzhu0823@gmail.com
# @File    : process.py
# @Software: PyCharm

import argparse
import numpy as np
from load_data import load_data


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', default='./fabric_data', type=str)
    parser.add_argument('-size', default=400, type=int)
    args = parser.parse_args()
    datas = load_data(args)
    for data in datas:
        process(data['img1'], data['img2'], data['info']['bbox'], data['label'])