# -*- coding: UTF-8 -*-
# filename: main date: 2018/4/17 10:25  
# author: FD
import numpy as np
from scipy import interpolate
import pandas as pd
from filterpy.kalman import KalmanFilter

from OneTagVerticalEKF import VerticalEKF
from OneTagRotatingEKF import RotatingEKF
from  dataprecess.ImageUtils import ImageUtils
import matplotlib.pyplot as plt

noises = {8005: 0.0419677551255, 8002: 0.0364688555131}


def getData(filepath):
    file_content = np.loadtxt(filepath, delimiter=',', skiprows=1)
    # EPC time RSSI phase
    data = file_content[:, [0, 3, 4, 5]]
    # 时间单位转化为毫秒
    data[:, 1] = data[:, 1] / 1000.0
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


def segment(data, window_size=10, threshold=0.03, consecutiveCount=15):
    activeIndexes = []
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
            activeIndexes += range(i - window_size, i)
    activeIndexes = list(set(range(data.size)) - set(activeIndexes))
    activeIndexes.sort()
    return activeIndexes


def rotatingBrokenRegion(data, RSSILowThreshold=-66, maxTimeThreshold=200):
    result = []
    shape = data.shape
    for i in range(1, shape[0]):
        if data[i, 1] < RSSILowThreshold and data[i - 1, 1] < RSSILowThreshold and data[i, 0] - data[
                    i - 1, 0] > maxTimeThreshold:
            result.append([data[i - 1, 0], data[i, 0]])
    return np.asarray(result)


def testPreprocess():
    filepath = unicode("../data/lab-2018-4-20/h2.csv", "utf8")
    data = getData(filepath)
    ImageUtils.draw_phase_diagram(filepath)
    preprocessedData = preprocess(data)
    rotatingRegion = rotatingBrokenRegion(data[:, [1, 2]])
    epcActiveDict = {}
    for epc in preprocessedData.keys():
        dataOfEpc = preprocessedData[epc][:, 1]
        # 通过 RSSI 判断是旋转还是垂直运动
        activeIndexes = segment(dataOfEpc)
        epcActiveDict[epc] = activeIndexes
        plt.plot(range(dataOfEpc.size), dataOfEpc.tolist(), label=str(epc))
        # plt.plot(range(activeIndexes.__len__()),activeIndexes,label=str(epc)+" KL")

    for epc in preprocessedData.keys():
        # 垂直上升运动
        dataOfEpc = preprocessedData[epc][:, 1]
        activeIndexes = epcActiveDict[epc]
        activePhaseData = dataOfEpc[activeIndexes]
        plt.figure()
        plt.title("active position")
        plt.plot(activeIndexes, activePhaseData)

        if (rotatingRegion.size == 0):
            verticalInitialState = np.array([0, 0, 0.1]).T
            verticalAntennaPos = np.asarray([[0, 1.0, 0.856]])
            verticalPhaseData = activePhaseData
            verticalEKF = VerticalEKF(verticalInitialState, verticalAntennaPos, verticalPhaseData)
            plt.figure()
            plt.title("vertical EKF result")
            result = verticalEKF.getResult()
            plt.plot(result)
            plt.show()
        else:
            # 圆周运动
            rotatingInitialState = np.array([0, 0, 0.1]).T
            rotatingAntennaPos = np.asarray([[0, 1.0, 0.856]])
            rotatingPhaseData = activePhaseData
            rotatingRadiuses = np.linspace(0.18, 0.6, 10)
            radiuses = []
            for rotatingRadius in rotatingRadiuses:
                # rotatingRadius=0.33
                rotatingEKF = RotatingEKF(rotatingInitialState, rotatingAntennaPos, rotatingPhaseData, rotatingRadius)
                result = rotatingEKF.getResult()
                radius = max(result)
                radiuses.append(radius)
            # 进行二项拟合
            z1 = np.polyfit(rotatingRadiuses, radiuses, 2)
            a, b, c = z1
            c=c - np.pi/2
            print z1
            delta = b ** 2 - 4 * a * c
            if (delta >= 0):
                trueRadius=(-b-np.sqrt(delta))/(2*a)
                print 'trueRadius= ',trueRadius

            p1 = np.poly1d(z1)
            yvals = p1(rotatingRadiuses)

            plt.figure()
            plt.plot(rotatingRadiuses, radiuses, '*', label='original values')
            plt.plot(rotatingRadiuses, yvals, 'r', label='polyfit values')
            plt.show()
            pass

    return


def main():
    testPreprocess()
    return


if __name__ == "__main__":
    main()
