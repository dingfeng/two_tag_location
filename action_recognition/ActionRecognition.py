# -*- coding: UTF-8 -*-
# filename: main date: 2018/4/17 10:25  
# author: FD
import numpy as np
from scipy import interpolate
import pandas as pd
from filterpy.kalman import KalmanFilter


def getData(filepath):
    file_content = np.loadtxt(filepath, delimiter=',', skiprows=1)
    # EPC time RSSI phase
    data = file_content[:, [0, 3, 4, 5]]
    return data


def interp1d(timeSeriesData, startTime, timeIndex, dataIndex):
    f_linear = interpolate.interp1d(timeSeriesData[:, timeIndex], timeSeriesData[:, dataIndex])
    time = timeSeriesData[:, 0]
    maxX = time[-1]
    minX = time[0]
    timeInterval = 10
    x = np.linspace(startTime, maxX, (maxX - minX) / timeInterval)
    y = f_linear(x)
    return y


# 使用标准 rts卡尔曼滤波进行平滑
def filter(data, var):
    rk = KalmanFilter(dim_x=1, dim_z=1)
    rk.F = np.array([[1., 1.],
                     [0., 1.]])
    initValue = data[0]
    # 初始位置 先设置为0
    rk.x = np.array([initValue, 0]).T
    # 测量误差
    rk.R *= var
    rk.P *= 10
    rk.Q *= 0.001
    mu, cov, _, _ = rk.batch_filter(data)
    M, P, C_ = rk.rts_smoother(mu, cov)
    return M


def preprocessOneData(startTime, oneData):
    # 对相位unwrap
    timeSeriesData = oneData[oneData[:, 0].argsort()]
    timeSeriesData[:, 2] = np.unwrap(timeSeriesData[:, 2])
    # 时间单位转化为毫秒
    timeSeriesData[:, 0] = timeSeriesData[:, 0] / 1000
    # 对相位进行插值
    phase_y = interp1d(timeSeriesData, startTime, 0, 2)
    # 对RSSI进行插值
    rssi_y = interp1d(timeSeriesData, startTime, 0, 1)
    # 对phase_y 进行kalman filter
    phase_y = filter(phase_y)
    # 对rssi_y 进行kalman filter
    window_size = 12
    rssi_y = pd.rolling_mean(rssi_y, window=window_size)[window_size - 1:];
    result = np.array([phase_y, rssi_y]).T
    return result


# 预处理 相位unwrap、插值和滑动平均滤波
def preprocess(data):
    # interpolate
    epcs = np.unique(data[:, 0])
    maxStartTime = -1
    result = {}
    for epc in epcs:
        oneData = data[np.where(data[:, 0] == epc)[0], 1:]
        result[epc] = oneData
        oneDataStartTime = np.min(data[:, 0])
        maxStartTime = np.max([oneDataStartTime, maxStartTime])
    for epc in epcs:
        result[epc] = preprocess(result[epc])
    return result


def main():

    pass


if __name__ == "__main__":
    main()
