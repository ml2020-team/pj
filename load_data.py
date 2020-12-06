#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/12/6 3:09 下午
# @Author  : zbl
# @Email   : blzhu0823@gmail.com
# @File    : load_data.py
# @Software: PyCharm


import argparse
import os
import json
from os.path import join as pjoin
from PIL import Image

def load_data(args):
    datas = []
    exs = ['.jpg', '.json']
    data_path = args.data_path
    paths = os.listdir(pjoin(data_path, 'label_json'))
    for path in paths:
        prefixs = list(map(lambda s: s.split('.')[0], os.listdir(pjoin(data_path, 'label_json', path))))
        for prefix in prefixs:
            try:
                img1 = Image.open(pjoin(data_path, 'temp', path, prefix + exs[0]))
                img2 = Image.open(pjoin(data_path, 'trgt', path, prefix + exs[0]))
                info = json.load(open(pjoin(data_path, 'label_json', path, prefix + exs[1]), encoding='utf-8'))
            except:
                continue
            datas.append({'img1': img1, 'img2': img2, 'info': info})
    return datas

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', default='./fabric_data', type=str)
    args = parser.parse_args()
    datas = load_data(args)

