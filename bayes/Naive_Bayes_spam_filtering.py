# -*- encoding:utf-8 -*-

import re
import random
import numpy as np

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
        
# 文件解析
def textParse(bigString):
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2] # 去掉少于两个字符的字符串

# 垃圾邮件测试函数
def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1,26): # 各25个文件
        # 导入并解析文本文件
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocbList(docList) # 创建词汇表
    trainingSet = range(50) # 一共50个文件
    testSet = []
    # 随机构建训练集,剩余部分作为测试集,留存交叉验证（hold-out cross validation）
    for i in range(10):
        # uniform() 方法将随机生成下一个实数，它在[x,y]范围内。
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0v, p1v, pSpam = trainNB0(np.array(trainMat), array(trainClasses))
    errorCount = 0
    # 对测试集分类
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error",docList[docIndex])
    print('the error rate is: ',float(errorCount)/len(testSet))