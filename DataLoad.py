# !/usr/bin/env python
# -*- coding:utf-8 -*-　
# Author: leven
# File: DataLoad.py
# Date: 2022/1/19 0019

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')


def distribution_test(train, test, features):
    """
    Test the distribution of train data and test data is same or not and plot them.
    :param train: dataframe of train data.
    :param test: dataframe of test data.
    """
    plt.figure(figsize=(10, 38 * 6))
    for feature in features:
        sns.distplot(train[feature])
        sns.distplot(test[feature])
        plt.title(feature)
        plt.show()


def read_data(path):
    """
    Read train data and test data from txt file.
    :param path: file path
    :return: train dataframe and test dataframe
    """
    df_train = pd.read_csv(path+'zhengqi_train.txt', sep='\t')
    df_test = pd.read_csv(path+'zhengqi_test.txt', sep='\t')
    feature_list = list(df_train.columns)       # Get all columns name and make them into list
    feature_list.remove('target')               # Remove the column of target
    # distribution_test(df_train, df_test, feature_list)
    feature_list = ['V0', 'V1', 'V3', 'V4', 'V8', 'V10', 'V12', 'V15', 'V16', 'V18', 'V24', 'V25', 'V26', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V36']
    df_train = df_train[['V0', 'V1', 'V3', 'V4', 'V8', 'V10', 'V12', 'V15', 'V16', 'V18', 'V24', 'V25', 'V26', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V36', 'target']]
    df_test = df_test[feature_list]
    # print(df_train.corr())
    feature_list = ['V0', 'V1', 'V3', 'V4', 'V8', 'V12', 'V16', 'V31']
    df_train = df_train[['V0', 'V1', 'V3', 'V4', 'V8', 'V12', 'V16', 'V31', 'target']]
    df_test = df_test[feature_list]

    return df_train, df_test, feature_list


def data_split(dfm, feature_list):
    train_x_, val_x_, train_y_, val_y_ = train_test_split(dfm[feature_list], dfm['target'], test_size=0.2, random_state=1)
    train_x_ = train_x_.reset_index(drop=True)
    train_y_ = train_y_.reset_index(drop=True)
    val_x_ = val_x_.reset_index(drop=True)
    val_y_ = val_y_.reset_index(drop=True)

    train_x_ = np.array(train_x_)
    train_y_ = np.array(train_y_)
    val_x_ = np.array(val_x_)
    val_y_ = np.array(val_y_)
    return train_x_, train_y_, val_x_, val_y_

