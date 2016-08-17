# -*- encoding:utf-8 -*-

import numpy as np

# 加载数据
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    # 1代表侮辱性文字，0代表正常言论
    return postingList,classVec

# 创建词汇表
def createVocabList(dataSet):
    # 创建一个空集
    vocabSet = set([])
    for document in dataSet:
        # 创建两个集合的并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

# 从文本中构建词向量,词集模型（set-of-words model）
def setOfWords2Vec(vocabList, inputSet):
    # 创建一个其中所含元素都为0的向量
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

# 词袋模型（bag-of-words model）
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

# 从词向量计算概率p(c)、p(w|c)
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 初始化概率
    # 计算p(w0|1)p(w1|1)p(w2|1)时，
    # 如果其中一个概率值为0，那么最后的乘积也为0。
    # 为降低这种影响，可以将所有词的出现数初始化为1，并将分母初始化为2。
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 向量相加
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # p1Vect = p1Num / p1Denom
    # p0Vect = p0Num / p1Denom
    # 当计算乘积p(w0|ci)p(w1|ci)p(w2|ci)...p(wN|ci)时，
    # 由于大部分因子都非常小，所以程序会下溢出或者得到不正确的答案。
    # 通过求对数可以避免下溢出或者浮点数舍入导致的错误。
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p1Denom)
    return p0Vect, p1Vect, pAbusive

# 朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 元素相乘
    p1 = np.sum(vec2Classify * p1Vec) + log(pClass1) # p(w|c)p(c),对数有ln(a*b) = ln(a)+ln(b), 
    p0 = np.sum(vec2Classify * p0Vec) + log(1.0 - pClass1) #　p(w)对一篇文档来说是固定的所以不用算
    if p1 > p0:
        return 1
    else:
        return 0

# 测试函数
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
