# !/usr/bin/env python
# -*- coding:utf-8 -*-　
# Author: leven
# File: main.py
# Date: 2022/1/19 0019

from DataLoad import read_data, data_split

if __name__ == '__main__':
    path = './data/'
    train, test, features = read_data(path)
    train_x, train_y, val_x, val_y = data_split(train, features)