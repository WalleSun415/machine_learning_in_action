# coding=utf-8
import numpy as np


def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat


def distEclud(vecA, vecB):
    return np.sqrt(np.sum(pow(vecA.A - vecB.A, 2)))


def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        maxJ = max(dataSet[:, j])
        rangeJ = float(maxJ - minJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)  # 随机生成k个0-1.0之间的随机数作为第j个特征的中心取值
    return centroids


def kMeans(dataSet, k, disMeas=distEclud, createCent=randCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf; minIndex = -1
            for j in range(k):
                distJI = disMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True  # 所有样本均未改变所属簇类别，则不再while循环，表示已收敛
            clusterAssment[i, :] = minIndex, minDist**2
        print centroids
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]  # np.nonzero()[0]返回索引值，np.nonzero()[0]返回对应值
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment


# 二分K-均值算法(bisecting K-means)
def biKmeans(dataSet, k, distMeas=distEclud):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j, 1] = distMeas(dataSet[j, :], np.mat(centroid0))
    while len(centList) < k:
        lowestSSe = np.inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0], :]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = np.sum(splitClustAss[:, 1])
            sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print("sseSplit and notSplit: ", sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSe:
                bestCentoSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSe = sseSplit + sseNotSplit
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)  # 重新修改簇编号以划分簇
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentoSplit  # 重新修改簇编号以划分簇
        print("the bestCentToSplit is: ", bestCentoSplit)
        print ("the len of bestClustAss is: ", len(bestClustAss))
        centList[bestCentoSplit] = bestNewCents[0, :]
        centList.append(bestNewCents[1, :])
        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentoSplit)[0], :] = bestClustAss
    return centList, clusterAssment


if __name__ == '__main__':
    # datMat = np.mat(loadDataSet('testSet.txt'))
    # print(min(datMat[:, 0]), max(datMat[:, 0]))
    # print(min(datMat[:, 1]), max(datMat[:, 1]))
    # print(randCent(datMat, 2))
    # print(distEclud(datMat[0], datMat[1]))

    # datMat = np.mat(loadDataSet('testSet.txt'))
    # myCentroids, clusterAssing = kMeans(datMat, 4)
    # print(myCentroids, clusterAssing)

    datMat3 = np.mat(loadDataSet('testSet2.txt'))
    centList, myNewAssments = biKmeans(datMat3, 3)
    print(centList, myNewAssments)