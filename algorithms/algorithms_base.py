from abc import ABCMeta, abstractmethod
import numpy as np

class MlAlgorithms(metaclass=ABCMeta):
    """
    算法抽象基类，规范通用接口
    """
    @abstractmethod
    def fit(self, inputs: np.array, labels: np.array):
        """
        模型训练
        :param inputs: 训练数据的特征矩阵
        :param labels: 训练数据的标签array
        :return:
        """
        pass

    @abstractmethod
    def predict(self, inputs: np.array):
        """
        模型推理
        :param inputs: 推理数据的特征矩阵
        :return: 推理出的标签array
        """
        pass