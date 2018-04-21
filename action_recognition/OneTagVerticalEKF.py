# -*- coding: UTF-8 -*-
# filename: OneD date: 2018/4/15 18:58
# author: FD
# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy.random as random
from numpy.random import randn
from math import sqrt
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from numpy import array, eye, asarray
from dataprecess.FileReader import FileReader
import dataprecess.FDUtils as FDUtils
from dataprecess.ImageUtils import ImageUtils

''' 扩展卡尔曼滤波'''


class VerticalEKF:
    '''相位部分'''
    frequency = 920.625e6
    wave_length = float(3e8 / frequency)
    distances = None
    # antenna_pos = asarray([[0, 1, 0.8]])
    antenna_pos = None
    rk = None
    initialState = None
    phaseData = None

    def __init__(self, initialState, antenna_pos, phaseData):
        self.initialState = initialState
        self.antenna_pos = antenna_pos
        self.phaseData = phaseData
        self.distances = asarray([np.sqrt(antenna_pos[0, 0] ** 2 + antenna_pos[0, 1] ** 2 + antenna_pos[0, 2] ** 2)])
        pass

    def H_of(self, x, antenna_pos):
        pos_z = float(x[0])
        shape = antenna_pos.shape
        result = np.zeros((shape[0], 3))
        result[0][0] = (pos_z - antenna_pos[0, 2]) / sqrt(
            (antenna_pos[0, 0]) ** 2 + (antenna_pos[0, 1]) ** 2 + (pos_z - antenna_pos[0, 2]) ** 2)
        return result

    def hx(self, x, antenna_pos):
        pos_z = float(x[0])
        result = []
        result.append(sqrt(
            antenna_pos[0, 0] ** 2 + antenna_pos[0, 1] ** 2 + (pos_z - antenna_pos[0, 2]) ** 2))
        return asarray(result)

    def get_rk(self):
        self.rk = ExtendedKalmanFilter(dim_x=3, dim_z=1)
        dt = 0.01
        self.rk.F = asarray([[1., dt, dt * dt / 2],
                             [0., 1., dt],
                             [0., 0., 1.],
                             ])
        # self.rk.x = array([0, 0, 0.1]).T
        self.rk.x = self.initialState
        # 测量误差
        self.rk.R *= 0.01
        self.rk.P *= 10
        self.rk.Q *= 0.0001
        return self.rk

    # 单位为cm
    def get_distance_by_phase(self, phase):
        distance = self.wave_length * phase / (4 * np.pi)
        return distance

    # 根据相位差获得距离差
    def get_delta_d(self, delta_phase):
        delta_d = None
        if np.abs(delta_phase) < np.pi:
            delta_d = delta_phase * self.wave_length / (4 * np.pi)
        elif delta_phase >= np.pi:
            delta_d = (delta_phase - 2 * np.pi) * self.wave_length / (4 * np.pi)
        elif delta_phase < -np.pi:
            delta_d = (delta_phase + 2 * np.pi) * self.wave_length / (4 * np.pi)
        return delta_d

    # return start[0] list
    def getResult(self):
        rk = self.get_rk()
        rk.predict_update(self.distances, self.H_of, self.hx, args=self.antenna_pos, hx_args=self.antenna_pos)
        predicted = []
        for i in range(1, self.phaseData.size):
            distance = self.get_delta_d(self.phaseData[i] - self.phaseData[i - 1])
            self.distances += asarray([distance])
            rk.predict_update(self.distances, self.H_of, self.hx, args=self.antenna_pos, hx_args=self.antenna_pos)
            predicted.append(rk.x[0])
        return predicted

    # generate simulation data
    def get_simulation_data(self):
        phases = []
        z = np.linspace(0, 0.1, 100)
        for i in range(z.size):
            distance = [
                sqrt(self.antenna_pos[0, 0] ** 2 + self.antenna_pos[0, 1] ** 2 + (z[i] - self.antenna_pos[0, 2]) ** 2)]
            if i == 0:
                self.distances = distance
            phase = [4 * np.pi * distance[0] / self.wave_length + np.random.normal(0, 0.01, 1)[0]]
            phases.append(phase)
        return asarray(phases)

    # start simulation process
    def start_simulation(self):
        simulationData = self.get_simulation_data()[:, 0]
        plt.figure()
        plt.title("phase")
        plt.plot(simulationData)
        plt.show()
        rk = self.get_rk()
        rk.predict_update(self.distances, self.H_of, self.hx, args=self.antenna_pos, hx_args=self.antenna_pos)
        predicted = []
        for i in range(1, simulationData.size):
            distance = self.get_delta_d(simulationData[i] - simulationData[i - 1])
            self.distances += asarray([distance])
            rk.predict_update(self.distances, self.H_of, self.hx, args=self.antenna_pos, hx_args=self.antenna_pos)
            predicted.append(rk.x[0])
        plt.figure()
        plt.title("z move")
        plt.plot(predicted)
        plt.show()

# def main():
#     verticalEKF = VerticalEKF()
#     verticalEKF.start_simulation()
#     return
#
#
# if __name__ == "__main__":
#     main()
