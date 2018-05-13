# -*- coding: UTF-8 -*-
# filename: Simulation date: 2018/5/10 13:09
# author: FD
from numpy import array, eye, asarray
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd
from filterpy.kalman import KalmanFilter
from matplotlib import rc
from OneTagVerticalEKF import VerticalEKF
from OneTagRotatingEKF import RotatingEKF
rc('text', usetex=True)
frequency = 920.625e6
wave_length = float(3e8 / frequency)
antenna = np.array([0, 1.0, 0.856])

distances = []
groundtruths = None


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


def rotatingBrokenRegion(data, RSSILowThreshold=-66, maxTimeThreshold=200):
    result = []
    shape = data.shape
    for i in range(1, shape[0]):
        if data[i, 1] < RSSILowThreshold and data[i - 1, 1] < RSSILowThreshold and data[i, 0] - data[
                    i - 1, 0] > maxTimeThreshold:
            result.append([data[i - 1, 0], data[i, 0]])
    return np.asarray(result)


def get_simulation_data(r, v):
    global distances
    global groundtruths
    distances = []
    theta = np.pi / 2
    pointCount = theta * 100 / v
    thetas = np.linspace(0, theta, pointCount)
    groundtruths = np.hstack((thetas, thetas[::-1]))
    phases = []
    for i in range(thetas.size):
        # 计算x坐标
        tagZ = r * (1.0 - np.cos(thetas[i]))
        tagX = r * np.sin(thetas[i])
        distance = np.sqrt(tagX ** 2 + 1.0 + (tagZ - 0.856) ** 2)
        distances.append(distance)
        phase = [4 * np.pi * distance / wave_length + np.random.normal(0, 0.3, 1)[0]]  # % (2 * np.pi)
        phases.append(phase)
    phases = np.asarray(phases)
    reversedPhases = phases[::-1]
    phases = np.vstack((phases, reversedPhases))
    return asarray(phases)


def main():
    global groundtruths
    rs=np.linspace(0.2,0.4,10)
    vs=np.linspace(0.5,np.pi/2,10)
    errors = []
    for r in rs:
        for v in vs:
             for i in range(10):
                simulationData = get_simulation_data(r, v)
                filteredSimulationData = filter(simulationData, 0.3)
                rotatingInitialState = np.array([0, 0, 0.1]).T
                rotatingAntennaPos = np.asarray([[0, 1.0, 0.856]])
                rotatingRadiuses = np.linspace(0.18, 0.6, 10)
                radiuses = []
                for rotatingRadius in rotatingRadiuses:
                    rotatingEKF = RotatingEKF(rotatingInitialState, rotatingAntennaPos, filteredSimulationData, rotatingRadius)
                    result = rotatingEKF.getResult()
                    radius = max(result)
                    radiuses.append(radius)
                # 进行二项拟合
                z1 = np.polyfit(rotatingRadiuses, radiuses, 2)
                a, b, c = z1
                c = c - np.pi / 2
                delta = b ** 2 - 4 * a * c
                if (delta >= 0):
                    trueRadius = (-b - np.sqrt(delta)) / (2 * a)
                    error=np.abs(trueRadius-r) / trueRadius
                    print trueRadius,r
                    errors.append(error)
    print str(errors)
    return


if __name__ == '__main__':
    main()
