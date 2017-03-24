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

# without kernel function
# class optStruct:
#     def __init__(self, dataMatIn, classLabels, C, toler):
#         self.X = dataMatIn
#         self.labelMat = classLabels
#         self.C = C
#         self.tol = toler
#         self.m = shape(dataMatIn)[0]
#         self.alphas = mat(zeros((self.m, 1)))
#         self.b = 0
#         self.eCache = mat(zeros((self.m, 2)))



def calcEk(oS, k):
    """
    without kernel function
    """
    # fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
    """
    with kernel function
    """
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k]) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJ(i, oS, Ei):
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i: continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)  # 选择使|E1-E2|最大的α2
            if deltaE > maxDeltaE:
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
        return j, Ej


def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy()
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H: print("L==H");return 0
        # eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T  # without kernel functon
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]  # with kernel function
        if eta >= 0: print("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if abs(oS.alphas[j] - alphaJold) < 0.00001:  # 判断α2是否有足够大的变化
            print("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[i] * oS.labelMat[j] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)

        # without kernel function
        # b1 = oS.b - Ei - oS.labelMat[i] * oS.X[i, :] * oS.X[i, :].T * (oS.alphas[i] - alphaIold) - \
        #      oS.labelMat[j] * oS.X[j, :] * oS.X[i, :].T * (oS.alphas[j] - alphaJold)
        # b2 = oS.b - Ej - oS.labelMat[i] * oS.X[i, :] * oS.X[j, :].T * (oS.alphas[i] - alphaIold) - \
        #      oS.labelMat[j] * oS.X[j, :] * oS.X[j, :].T * (oS.alphas[j] - alphaJold)

        # with kernel function
        b1 = oS.b - Ei - oS.labelMat[i] * oS.K[i, i] * (oS.alphas[i] - alphaIold) - \
             oS.labelMat[j] * oS.K[j, i] * (oS.alphas[j] - alphaJold)
        b2 = oS.b - Ej - oS.labelMat[i] * oS.K[i, j] * (oS.alphas[i] - alphaIold) - \
             oS.labelMat[j] * oS.K[j, j] * (oS.alphas[j] - alphaJold)
        if (oS.alphas[i] > 0) and (oS.alphas[i] < oS.C): oS.b = b1
        elif (oS.alphas[j] > 0) and (oS.alphas[j] < oS.C): oS.b = b2
        else: oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


# with kernel function
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
    iterNum = 0
    entireSet = True; alphaPairsChanged = 0
    while (iterNum < maxIter) and (alphaPairsChanged > 0 or entireSet):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iterNum: %d, i: %d, pairs changed %d" % (iterNum, i, alphaPairsChanged))
            iterNum += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iterNum: %d, i: %d, pairs changed %d" % (iterNum, i, alphaPairsChanged))
            iterNum += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        print("iteration: %d" % iterNum)
    return oS.b, oS.alphas


def calcWs(alphas, dataArr, labelArr):
    X = mat(dataArr); labelMat = mat(labelArr).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


def kernelTrans(X, A, kTup):
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(-K / (kTup[1] ** 2))
    else:
        raise NameError("Houston We Have a Problem -- That Kernel is not recognized")
    return K


def testRbf(k1=1.3):
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    datMat = mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % shape(sVs)[0])
    m, n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(alphas[svInd], labelSV) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))
    dataArr, labelArr = loadDataSet('testSetRBF2.txt')
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(alphas[svInd], labelSV) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels


def testDigits(kTup=('rbf', 10)):
    dataArr, labelArr = loadImages('./digits/trainingDigits')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat = mat(dataArr);
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % shape(sVs)[0])
    m, n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * multiply(alphas[svInd], labelSV) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))
    dataArr, labelArr = loadImages('./digits/testDigits')
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * multiply(alphas[svInd], labelSV) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))


if __name__ == '__main__':
    # dataArr, labelArr = loadDataSet('testSet.txt')
    # print(dataArr)
    # print(labelArr)
    # b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    # b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
    # print(b)
    # print(alphas[alphas > 0])
    # ws = calcWs(alphas, dataArr, labelArr)
    # print(ws)
    #testRbf()
    testDigits(('rbf', 10))
