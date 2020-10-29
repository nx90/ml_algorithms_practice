# coding: utf-8
import numpy as np

from algorithms.algorithms_base import MlAlgorithms


class LinearRegress(MlAlgorithms):
    def __init__(self):
        self.paras = None

    def fit(self, inputs: np.array, labels: np.array):
        # TODO: 维度检测，不正常则输出异常
        para_b = np.ones(inputs.shape[0])
        inputs = np.column_stack((inputs, para_b))
        inputs_transpose = inputs.transpose()
        self.paras = np.linalg.inv(inputs_transpose.dot(inputs)).dot(inputs_transpose).dot(labels)

    def predict(self, inputs: np.array):
        para_b = np.ones(inputs.shape[0])
        inputs = np.column_stack((inputs, para_b))
        return inputs.dot(self.paras)
