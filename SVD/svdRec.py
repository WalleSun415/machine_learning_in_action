# coding=utf-8
import numpy as np


def loadExData():
    return[[1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [1, 1, 1, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1]]


if __name__ == '__main__':
    Data = loadExData()
    U, Sigma, VT = np.linalg.svd(Data)
    print(Sigma)
