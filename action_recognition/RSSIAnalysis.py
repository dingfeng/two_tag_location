# -*- coding: UTF-8 -*-
# filename: RSSIAnalysis date: 2018/4/20 20:54  
# author: FD
import numpy as np
from  dataprecess.ImageUtils import ImageUtils
import matplotlib.pyplot as plt


def getData(filepath):
    file_content = np.loadtxt(filepath, delimiter=',', skiprows=1)
    # EPC time RSSI phase
    data = file_content[:, [0, 3, 4, 5]]
    return data


def rotatingBrokenRegion(data, RSSILowThreshold=-66, maxTimeThreshold=200):
    result = []
    shape = data.shape
    for i in range(1, shape[0]):
        if data[i, 1] < RSSILowThreshold and data[i - 1, 1] < RSSILowThreshold and data[i, 0] - data[
                    i - 1, 0] > maxTimeThreshold:
            result.append([data[i - 1, 0], data[i, 0]])
    return np.asarray(result)


def main():
    filepath = unicode("../data/lab-2018-4-20/h1.csv", "utf8")
    data = getData(filepath)
    data[:, 1] = data[:, 1] / 1000
    print rotatingBrokenRegion(data[:, [1, 2]])
    plt.figure()
    plt.title("rssi")
    plt.plot(data[:, 1], data[:, 2])
    plt.show()
    return


if __name__ == "__main__":
    main()
