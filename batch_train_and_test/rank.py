import numpy as np


def mean_square_error(predict_label: np.array, real_label: np.array):
    # TODO: 维度限制，非正常维度直接报异常
    return (predict_label - real_label).transpose().dot((predict_label - real_label)) / predict_label.shape[0]
