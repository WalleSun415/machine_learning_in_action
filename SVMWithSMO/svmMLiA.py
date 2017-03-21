# coding=utf-8
from numpy import *


def loadDataSet(filename):
    dataMat = []; labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def selectJrand(i, m):  # 从m中随机取不等于i数值
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    b = 0; m, n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))
    iterNum = 0
    while iterNum < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix*dataMatrix[i, :].T)) + b  # 计算g(xi); multiply--矩阵点乘
            Ei = fXi - float(labelMat[i])  # 计算Ei
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b  # 计算g(xj)
                Ej = fXj - float(labelMat[j])  # 计算Ej
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if labelMat[i] != labelMat[j]:  # yi = yj时，计算L，H
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:  # yi ≠ yj时，计算L，H
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H: print("L==H"); continue  # continue表示本次循环结束，运行下一次for循环
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - \
                      dataMatrix[j, :] * dataMatrix[j, :].T  # 计算η
                if eta >= 0: print("eta>=0"); continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)  # 计算αj_new
                if abs(alphas[j] - alphaJold) < 0.00001: print("j not moving enough"); continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])  # 计算αi_new
                b1 = b - Ei - labelMat[i] * dataMatrix[i, :] * dataMatrix[j, :].T * (alphas[i] - alphaIold) - \
                     labelMat[j] * dataMatrix[i, :] * dataMatrix[j, :].T * (alphas[j] - alphaJold)  # 计算b1
                b2 = b - Ej - labelMat[i] * dataMatrix[i, :] * dataMatrix[j, :].T * (alphas[i] - alphaIold) - \
                     labelMat[j] * dataMatrix[j, :] * dataMatrix[j, :].T * (alphas[j] - alphaJold)  # 计算b2
                if (alphas[i] > 0) and (alphas[i] < C):
                    b = b1
                elif (alphas[i] > 0) and (alphas[i] < C):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("iter: %d i: %d, pairs changed %d" % (iterNum, i, alphaPairsChanged))
        if alphaPairsChanged == 0:
            iterNum += 1
        else:
            iterNum = 0
        print("iteration number: %d" % iterNum)
    return b, alphas


if __name__ == '__main__':
    dataArr, labelArr = loadDataSet('testSet.txt')
    print(dataArr)
    print(labelArr)
    b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    print(b)
    print(alphas[alphas > 0])
