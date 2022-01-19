# !/usr/bin/env python
# -*- coding:utf-8 -*-　
# Author: leven
# File: main.py
# Date: 2022/1/19 0019

import numpy as np
import pandas as pd
from DataLoad import read_data, data_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from sklearn.svm import SVR


def Train(train_x, train_y, val_x, val_y):
    model = []

    model_RF = RandomForestRegressor(n_estimators=145, random_state=0)
    model_RF.fit(train_x, train_y)
    model.append(model_RF)
    predict_RF = model_RF.predict(val_x)
    score_RF = mean_squared_error(predict_RF, val_y)
    print('RF: {}'.format(score_RF))

    model_KNN = KNeighborsRegressor(n_neighbors=4)
    model_KNN.fit(train_x, train_y)
    model.append(model_KNN)
    predict_KNN = model_KNN.predict(val_x)
    score_KNN = mean_squared_error(predict_KNN, val_y)
    print('KNN: {}'.format(score_KNN))

    model_LR = LinearRegression()
    model_LR.fit(train_x, train_y)
    model.append(model_LR)
    predict_LR = model_LR.predict(val_x)
    score_LR = mean_squared_error(predict_LR, val_y)
    print('LR: {}'.format(score_LR))

    model_light = LGBMRegressor()
    model_light.fit(train_x, train_y)
    model.append(model_light)
    predict_light = model_light.predict(val_x)
    score_light = mean_squared_error(predict_light, val_y)
    print('light: {}'.format(score_light))

    model_svm = SVR()
    model_svm.fit(train_x, train_y)
    model.append(model_svm)
    predict_svm = model_svm.predict(val_x)
    score_svm = mean_squared_error(predict_svm, val_y)
    print('svm: {}'.format(score_svm))

    predict = np.mean([predict_RF, predict_KNN, predict_LR, predict_light, predict_svm], axis=0)
    score = mean_squared_error(predict, val_y)
    print('mean: {}'.format(score))

    return model


def Predict(test, model, path):
    predict = []
    for mod in model:
        predict.append(mod.predict(test))
    predict = np.mean(predict, axis=0)
    print(predict)
    pd.Series(predict).to_csv(path+'result.txt', index=False)


if __name__ == '__main__':
    path = './data/'
    train, test, features = read_data(path)
    train_x, train_y, val_x, val_y = data_split(train, features)
    model = Train(train_x, train_y, val_x, val_y)
    test = np.array(test)
    Predict(test, model, path)