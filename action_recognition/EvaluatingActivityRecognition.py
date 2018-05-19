# -*- coding: UTF-8 -*-
# filename: EvaluatingActivityRecognition date: 2018/5/13 14:27  
# author: FD 
import numpy as np
import matplotlib.pyplot as plt
import os
import random as rand
RSSIThreshold = -66
maxAccuracy = 0.0
def getData(filepath):
    file_content = np.loadtxt(filepath, delimiter=',', skiprows=1)
    # EPC time RSSI phase
    data = file_content[:, [0, 3, 4, 5]]
    # 时间单位转化为毫秒
    data[:, 1] = data[:, 1] / 1000.0
    return data
def rotatingBrokenRegion(data, RSSILowThreshold=-66, maxTimeThreshold=0):
    global RSSIThreshold
    RSSILowThreshold=RSSIThreshold
    result = []
    shape = data.shape
    for i in range(1, shape[0]):
        if data[i, 1] < RSSILowThreshold and data[i - 1, 1] < RSSILowThreshold and data[i, 0] - data[
                    i - 1, 0] > maxTimeThreshold:
            result.append([data[i - 1, 0], data[i, 0]])
    return np.asarray(result)
filepath = unicode("../data/lab-2018-4-20/h2.csv", "utf8")


def isVertical(filepath):
    data = getData(filepath)
    rotatingRegion = rotatingBrokenRegion(data[:, [1, 2]])
    if (rotatingRegion.size == 0):
        return True
    else:
        return False

def getResultInDir(dir,motionType):
    result = []
    subDirs = ['fast','normal','slow']
    filePaths = []
    for subDir in subDirs:
        dirPath = unicode(dir + subDir, 'utf8')
        fileList = os.listdir(dirPath)
        for file in fileList:
            if file.__contains__('csv'):
                filepath = os.path.join(dirPath,file)
                filePaths.append(filepath)
    for i in range(100):
        index = rand.randint(0,filePaths.__len__()-1)
        filepath=filePaths[index]
        if (motionType == isVertical(filepath)):
            result.append(1)
        else:
            result.append(0)
    return result

def getAccurateRate():
    global maxAccuracy
    global RSSIThreshold
    rootDir = '../data/'
    verticalDir = rootDir + 'vertical/'
    verticalResult = getResultInDir(verticalDir, True)
    circularDir = rootDir + 'circular/'
    circularResult = getResultInDir(circularDir, False)
    finalResult = verticalResult + circularResult
    accuracy = 1.0 * np.sum(finalResult) / finalResult.__len__()
    if accuracy >= maxAccuracy:
        print 'accuracy=',accuracy,' RSSIThreshold= ',RSSIThreshold,' ',str(finalResult)
        maxAccuracy=accuracy
    return accuracy

def main():
    global  RSSIThreshold
    global maxAccuracy
    accurateRates = []
    thresholds = [-55 - i * 0.5  for i in range(40)]
    print thresholds[0],',',thresholds[-1]
    maxAccuracy = 0.0
    for threshold in thresholds:
        RSSIThreshold = threshold
        accurateRate = getAccurateRate()
        accurateRates.append(accurateRate)
    # plt.figure()
    # plt.scatter(thresholds,accurateRates)
    # plt.show()
    print thresholds
    print accurateRates
    plt.figure()
    plt.plot(thresholds, accurateRates)
    plt.xlabel('RSSI threshold')
    plt.ylabel('accuracy')
    fig=plt.gcf()
    fig.set_size_inches(3.3492706944445,3.3492706944445/2)
    plt.savefig("recognition-accuracy.pdf",
                 dpi=1000,
                bbox_inches='tight',)
    return

if __name__ == '__main__':
    main()
    pass
