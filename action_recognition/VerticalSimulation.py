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
rc('text', usetex=True)
frequency = 920.625e6
wave_length = float(3e8 / frequency)
antenna = np.array([0, 1.0, 0.856])

distances = []
groundtruths=None

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


def get_simulation_data(v):
    global distances
    global groundtruths
    distances=[]
    h = 0.5
    pointCount = h * 100 / v
    x = np.linspace(0, h, pointCount)
    groundtruths=np.hstack((x,x[::-1]))
    phases = []
    for i in range(x.size):
        distance = np.sqrt(1.0 + (x[i] - 0.856) ** 2)
        distances.append(distance)
        phase = [4 * np.pi * distance / wave_length + np.random.normal(0, 0.3, 1)[0]]  # % (2 * np.pi)
        phases.append(phase)
    phases=np.asarray(phases)
    reversedPhases= phases[::-1]
    phases=np.vstack((phases,reversedPhases))
    return asarray(phases)


def main():
    global groundtruths
    velocities = np.linspace(0.1,3,20)
    errors=[]
    for velocity in velocities:
        error = 0.0
        for j in range(10):
            simulationData = get_simulation_data(velocity)
            filteredSimulationData = filter(simulationData, 0.3)
            verticalInitialState = np.array([0, 0, 0.1]).T
            verticalAntennaPos = np.asarray([[0, 1.0, 0.856]])
            verticalPhaseData = filteredSimulationData
            verticalEKF = VerticalEKF(verticalInitialState, verticalAntennaPos, verticalPhaseData)
            result=verticalEKF.getResult()
            for i in range(result.__len__()):
                groundtruth=groundtruths[i+1]
                estimatedValue=result[i]
                error += np.abs(groundtruth-estimatedValue)
        error = error / (result.__len__()*10)
        errors.append(error)
    # plt.figure()
    # plt.plot(velocities.tolist(),errors)
    # plt.show()
    print 'x= ',str(velocities.tolist())
    print 'y= ',str(errors)
    return


if __name__ == '__main__':
    main()
