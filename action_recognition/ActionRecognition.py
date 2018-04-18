# -*- coding: UTF-8 -*-
# filename: main date: 2018/4/17 10:25  
# author: FD
import numpy as np
from scipy import interpolate
import pandas as pd
from filterpy.kalman import KalmanFilter
from  dataprecess.ImageUtils import ImageUtils
import matplotlib.pyplot as plt
import scipy

noises = {8001: 0.0157178001838, 8002: 0.0364688555131}


def getData(filepath):
    file_content = np.loadtxt(filepath, delimiter=',', skiprows=1)
    # EPC time RSSI phase
    data = file_content[:, [0, 3, 4, 5]]
    return data


def interp1d(timeSeriesData, startTime, timeIndex, dataIndex):
    time = timeSeriesData[:, timeIndex]
    f_linear = interpolate.interp1d(time, timeSeriesData[:, dataIndex])
    maxX = time[-1]
    timeInterval = 10
    x = np.linspace(startTime, maxX, num=int((maxX - startTime) / timeInterval))
    y = f_linear(x)

    return y


# 使用标准 rts卡尔曼滤波进行平滑
def filter(data, var):
    rk = KalmanFilter(dim_x=2, dim_z=1)
    rk.F = np.array([[1., 1.],
                     [0., 1.]])
    rk.H = np.array([[1., 0.]])
    initValue = data[0]
    # 初始位置 先设置为0
    rk.x = np.array([initValue, 0]).T
    # 测量误差
    rk.R *= var
    rk.P *= 10
    rk.Q *= 0.001
    mu, cov, _, _ = rk.batch_filter(data)
    M, P, C_ = rk.rts_smoother(mu, cov)
    return M[:, 0]


def preprocessOneData(epc, startTime, oneData):
    # 对相位unwrap
    timeSeriesData = oneData[oneData[:, 0].argsort(), :]
    timeSeriesData[:, 2] = np.unwrap(timeSeriesData[:, 2])
    # plt.show()
    # 时间单位转化为毫秒
    timeSeriesData[:, 0] = timeSeriesData[:, 0] / 1000
    # 对相位进行插值
    phase_y = interp1d(timeSeriesData, startTime, 0, 2)
    # plt.figure()
    # plt.plot(phase_y)
    # 对RSSI进行插值
    rssi_y = interp1d(timeSeriesData, startTime, 0, 1)
    # 对phase_y 进行kalman filter

    phase_y = filter(phase_y, noises[int(epc)])
    # 对rssi_y 进行kalman filter
    window_size = 2
    rssi_y = pd.rolling_mean(rssi_y, window=window_size)[window_size - 1:];

    minSize = np.min([phase_y.size, rssi_y.size])
    rssi_y = rssi_y[0:minSize]
    phase_y = phase_y[0:minSize]
    result = np.vstack((rssi_y, phase_y)).T
    return result


# 预处理 相位unwrap、插值和滑动平均滤波
def preprocess(data):
    # interpolate
    epcs = np.unique(data[:, 0])
    result = {}
    for epc in epcs:
        oneData = data[np.where(data[:, 0] == epc)[0], 1:]
        result[int(epc)] = oneData
    startTime = 100
    for epc in epcs:
        result[int(epc)] = preprocessOneData(int(epc), startTime, result[int(epc)])
    return result


def segment(data, window_size=10, threshold=0.015, consecutiveCount=15):
    activeIndexes=[]
    currentConsecutive = 0
    for i in range(window_size, data.size, window_size):
        windowData = data[i - window_size:i]
        windowStd = np.std(windowData)
        print windowStd
        if (windowStd < threshold):
            currentConsecutive += 1
        else:
            currentConsecutive = 0
        if (currentConsecutive > consecutiveCount):
            activeIndexes.append(i - window_size)
    return activeIndexes


def testPreprocess():
    filepath = unicode("../data/v2.csv", "utf8")
    data = getData(filepath)
    ImageUtils.draw_phase_diagram(filepath)
    preprocessedData = preprocess(data)
    plt.figure()
    plt.title("after preprocess")
    epcActiveDict = {}
    for epc in preprocessedData.keys():
        dataOfEpc = preprocessedData[epc][:, 1]
        activeIndexes = segment(dataOfEpc)
        epcActiveDict[epc] = activeIndexes
        plt.plot(range(dataOfEpc.size), dataOfEpc.tolist(), label=str(epc))
        # plt.plot(range(activeIndexes.__len__()),activeIndexes,label=str(epc)+" KL")
    plt.legend()
    plt.figure()
    plt.title("active indexes")
    for epc in preprocessedData.keys():
        dataOfEpc = preprocessedData[epc][:, 1]
        activeIndexes = epcActiveDict[epc]
        plt.scatter(activeIndexes, dataOfEpc[activeIndexes], label=str(epc)+"marked",marker="*")
    plt.legend()

    plt.show()
    return


def main():
    testPreprocess()
    return


if __name__ == "__main__":
    main()
