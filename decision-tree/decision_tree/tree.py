# -*- coding: utf-8 -*-

from math import log
import operator
import pickle

'''
创建分支的伪代码函数createBranch()如下所示：

检测数据集中的每个子项是否属于同一分类：
    If so return 类标签；
    Else
        寻找划分数据集的最好特征(ID3,计算每个特征值划分数据集获得的信息增益,获得信息增益最高的特征就是最好的选择)
        划分数据集
        创建分支节点
            for 每个划分的子集
                调用函数createBranch并增加返回结果到分支节点中
        return 分支节点
'''

def createDataSet():
    '''
    创建一个简单鱼鉴定数据集
    '''

    dataSet = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]

    labels = ['no surfacing', 'flippers']

    return dataSet, labels


def calcShannonEnt(dataSet):
    '''
    计算给定数据集的香农熵

    参数：
        dataSet: list, 数据集
            eg: [[1,1,'yes'], [1,0,'no']]

    返回值：
        shannonEnt: float, 香农熵

    步骤：
        1. 计算数据集中实例的总数
        2. 创建一个数据字典，每个键值都记录了当前类别出现的次数
        3. 使用所有类别标签的发生频率计算类别出现的概率，使用这个概率计算香农熵
    '''

    numEntries = len(dataSet)
    labelCounts = {}
    # 为所有可能分类创建字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)

    return shannonEnt


def splitDataSet(dataSet, axis, value):
    '''
    按照给定特征划分数据集

    参数：
        dataSet: list, 待划分的数据集
            eg: [[1,1,'yes'], [1,0,'no']]

        axis: int, 划分数据集的特征

        value: 该划分特征的值

    返回值：
        eg:
            >>> myDat
            [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
            >>> tree.splitDataSet(myDat, 0, 1)
            [[1, 'yes'], [1, 'yes'], [0, 'no']]
    '''

    # 创建新的list对象
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 抽取
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)

    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    '''
    选择最好的数据集划分方式

    步骤：
        1. 计算整个数据集的原始香农熵
        2. 第一个for循环遍历数据中的所有特征
            3. 遍历当前特征中的所有唯一属性值，
               对每个特征划分一次数据集，
               然后计算该子数据集的新熵值，
               并对所有唯一特征值得到的熵求和
            4. 比较信息增益
        5. 返回最好特征划分的索引值
    '''

    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    # 遍历每个特征
    for i in range(numFeatures):
        # 将数据集中所有第i个特征所有可能存在的值写入这个新列表中
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        # 计算每种划分方式的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            # 计算最好的信息增益
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature


def majorityCnt(classList):
    '''
    返回classList中的多数类

    递归函数的第二个停止条件是使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组.
    这时无法简单地返回唯一的类标签，需要使用该函数挑选出现次数最多的类别作为返回值。
    '''

    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    '''
    递归构建决策树

    递归结束的条件是：程序遍历完所有划分数据集的属性，或者每个分支下的所有实例都具有相同的分类。
    如果所有实例具有相同的分类，则得到一个叶子节点或者终止块。
    '''

    classList = [example[-1] for example in dataSet]
    # 类别完全相同则停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 遍历完所以特征时返回出现次数最多的
    if len(dataSet[0]) == 1: # 1是因为还剩最后的类标号，此时dataSet像：[['yes'], ['no']]
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    # 将标签字符串转换为索引
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree, filename):
    '''
    使用pickle模块存储决策树
    '''

    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)


def grabTree(filename):
    with open(filename, 'rb') as fr:
        tree = pickle.load(fr)
    return tree