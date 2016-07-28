# -*- encoding:utf-8 -*-

from numpy import *
import operator
from os import listdir


def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 行数
    # 距离计算
    diffMat = tile(inX, (dataSetSize,1))-dataSet    # 输入向量与训练集中各实例的差        
    sqDiffMat = diffMat**2  # diffMat各个元素的平方
    sqDistances = sqDiffMat.sum(axis=1) # 默认的axis=0是普通的相加，axis=1是将一个矩阵的每一行向量相加
    distances = sqDistances**0.5    # 欧氏距离
    sortedDistIndicies = distances.argsort()    # argsort函数返回的是数组值从小到大的索引值
    classCount = {}
    # 选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    # 排序
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True) # classCount.iteritems()元组形式，itemgetter(1)表示按每个元组的第二个元素排序，返回结果类似[('c', 3), ('b', 2), ('a', 1)]
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    # 得到文件行数
    numberOfLines = len(fr.readlines())
    # 创建返回的Numpy矩阵
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    # 解析文件数据到列表
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    # 特征值相除
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.10  # 10%数据用于测试集
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if(classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
    print errorCount


def classifyPerson():
    resultList = ['not at all','in small doses', 'in large doses']
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print "You will probably like this person: ", resultList[classifierResult - 1]


def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    # 获取目录内容（即目录里的文件名），返回结果类似['9_97.txt', '9_98.txt', '9_99.txt']
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)   # 目录里的文件数
    trainingMat = zeros((m,1024))
    # 从文件名解析分类数字
    for i in range(m):
        fileNameStr = trainingFileList[i]   # 9_97.txt
        fileStr = fileNameStr.split('.')[0]     #take off .txt,返回结果类似9_97
        classNumStr = int(fileStr.split('_')[0])    # 9
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))