import csv
import random
import math
import operator

def loadDataSet(filename, split, trainingSet=[] , testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataSet = list(lines)
        for i in range(len(dataSet)-1):
            for j in range(4):
                dataSet[i][j] = float(dataSet[i][j])
            if random.random() < split:
                trainingSet.append(dataSet[i])
            else:
                testSet.append(dataSet[i])

# Test this function out with our iris dataset,as follows:
# trainingSet = []
# testSet = []
# loadDataSet('iris.data', 0.66, trainingSet, testSet)
# print('Train:' + str(len(trainingSet)))
# print('Test:' + str(len(testSet)))


def euclideanDistance(instance1, instance2, length):
    distance = 0.0
    for i in range(length):
        distance += pow(instance1[i] - instance2[i], 2)
    return math.sqrt(distance)

# Test this function with some sample data,as follows:
# data1 = [2, 2, 2, 'a']
# data2 = [4, 4, 4, 'b']
# distance = euclideanDistance(data1, data2, 3)
# print('Distance' + str(distance))


def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for i in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[i], length)
        distances.append((trainingSet[i], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

# Test out this function as follows:
# trainingSet = [[2, 2, 2, 'a'], [4, 4, 4, 'b']]
# testInstance = [5, 5, 5]
# neighbors = getNeighbors(trainingSet, testInstance, 1)
# print(neighbors)


def getResponse(neighbors):
    classVotes = {}
    for i in range(len(neighbors)):
        response = neighbors[i][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

# Test out this function with some test neighbors,as follows:
# neighbors = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
# response = getResponse(neighbors)
# print(response)


def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0

# Test this function with a test dataset and predictions,as follows:
# testSet = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
# predictions = ['a', 'a', 'a']
# accuracy = getAccuracy(testSet, predictions)
# print(accuracy)


def main():
    # prepare data
    trainingSet=[]
    testSet=[]
    split = 0.67
    loadDataSet('iris.data', split, trainingSet, testSet)
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))
    # generate predictions
    predictions=[]
    k = 3
    for i in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[i], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + str(result) + ', actual=' + str(testSet[i][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + str(accuracy) + '%')
    
main()