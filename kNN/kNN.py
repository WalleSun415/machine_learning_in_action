# coding=utf-8
from numpy import *
import operator


'''kNN应用-约会对象识别'''


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # tile重复输入数据，格式为(dataSetSize, 1)
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance ** 0.5
    sortedDistIndicies = distance.argsort()
    classCount = {}
    for i in range(k):
        voteInlabel = labels[sortedDistIndicies[i]]
        classCount[voteInlabel] = classCount.get(voteInlabel, 0) + 1
    sortClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortClassCount[0][0]


def file2Matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()  # 去除空格字符
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))  # NumPy中，/为矩阵值得点除，某些库中表示矩阵除法
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.20
    datingDataMat, datingLabels = file2Matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print "the total error rate is %f" % (errorCount/float(numTestVecs))


def classifyPerson():
    resultList = ['not at all', 'in small dose', 'in large dose']
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2Matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print("You will probably like this person:", resultList[classifierResult - 1])

'''手写识别系统'''
from os import listdir


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('digits/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for j in range(mTest):
        fileNameStr = testFileList[j]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
    print("the total number of errors is: %d" % errorCount)
    print("the total error rate is: %f" % (errorCount/float(mTest)))


if __name__ == '__main__':

    import kNN
    # datingDataMat, datingLabels = kNN.file2Matrix('datingTestSet2.txt')
    # normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
    #
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(121)  # 在1行1列，从上到下、左到右第1块子图上绘图
    # bx = fig.add_subplot(122)
    # ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0*array(datingLabels), 15.0*array(datingLabels))
    # bx.scatter(normMat[:, 0], normMat[:, 1], 15.0*array(datingLabels), 15.0*array(datingLabels))  # 15.0为点的半径
    # plt.show()

    #kNN.datingClassTest()
    #kNN.classifyPerson()
    #testVector = kNN.img2vector('digits/testDigits/0_13.txt')
    kNN.handwritingClassTest()