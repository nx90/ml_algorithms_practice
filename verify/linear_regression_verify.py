# coding: utf-8
import time

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from algorithms.linear_regression.linear_regression_core import LinearRegress
from batch_train_and_test import rank

def main():
    data_diabetes = load_diabetes()
    data = data_diabetes['data']
    target = data_diabetes['target']
    feature_names = data_diabetes['feature_names']
    train_X, test_X, train_Y, test_Y = train_test_split(data, target, train_size=0.8)

    time_start = time.time()
    model = LinearRegress()
    model.fit(train_X, train_Y)
    predict_Y = model.predict(test_X)
    time_end = time.time()
    print("自己的模型用时:", time_end - time_start)
    print("自己的预测:", predict_Y)
    print("自己的预测用自己的均方误差:", rank.mean_square_error(predict_Y, test_Y))
    print("自己的预测用sklearn的均方误差:", mean_squared_error(predict_Y,test_Y))

    time_start = time.time()
    model2 = LinearRegression()
    model2.fit(train_X, train_Y)
    predict2_Y = model2.predict(test_X)
    time_end = time.time()
    print("sklearn的模型用时:", time_end - time_start)
    print("sklearn的预测:", predict2_Y)
    print("真实标签:", test_Y)
    print("sklearn的预测用自己的均方误差:", rank.mean_square_error(predict2_Y, test_Y))
    print("sklearn的预测用sklearn的均方误差:", mean_squared_error(predict2_Y,test_Y))

if __name__ == '__main__':
    main()