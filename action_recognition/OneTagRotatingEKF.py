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


class RotatingEKF:
    distances = None
    antenna_pos = asarray([[0, 1., 1.0]])
    filepath = unicode("../data/active_V02.csv", "utf8")
    radius = None
    rk = None
    frequency = 920.625e6
    wave_length = float(3e8 / frequency)


    def __init__(self):
        pass

    def H_of(self, x, antenna_pos):
        r = self.radius
        theta = float(x[0])
        shape = antenna_pos.shape
        result = np.zeros((shape[0], 3))
        denominator = np.sqrt(
            (r * np.sin(theta) - antenna_pos[0, 0]) ** 2 + antenna_pos[0, 1] ** 2 + (
                r * (1 - np.cos(theta)) - antenna_pos[0, 2]) ** 2)
        result[0][0] = r * ((r - antenna_pos[0, 2]) * np.sin(theta) - antenna_pos[0, 0] * np.cos(theta)) / denominator
        return result

    def hx(self,x, antenna_pos):
        result = []
        r = self.radius
        theta = float(x[0])
        x = r * np.sin(theta)
        z = r * (1 - np.cos(theta))
        result.append(sqrt(
            (x - antenna_pos[0, 0]) ** 2 + antenna_pos[0, 1] ** 2 + (z - antenna_pos[0, 2]) ** 2))
        return asarray(result)

    def get_rk(self):
        self.rk = ExtendedKalmanFilter(dim_x=3, dim_z=1)
        dt = 0.01
        self.rk.F = asarray([
            [1., dt, dt * dt / 2],
            [0., 1., dt],
            [0., 0., 1.]
        ])
        # 初始位置 先设置为0
        self.rk.x = array([0, 0, 0.2]).T
        # 测量误差
        self.rk.R *= 0.01
        self.rk.P *= 10
        self.rk.Q *= 0.001
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

    def getIdealPhase(self, theta):
        r = self.radius
        x = r * np.sin(theta)
        z = r * (1 - np.cos(theta))
        distance = [sqrt((x - self.antenna_pos[0, 0]) ** 2 + self.antenna_pos[0, 1] ** 2 + (z - self.antenna_pos[0, 2]) ** 2)]
        phase = 4 * np.pi * distance[0] / self.wave_length % (2 * np.pi)
        return phase

    def get_simulation_data(self):
        phases = []
        thetas = np.linspace(0, np.pi / 2, 1000)
        print thetas
        r = 0.4
        for i in range(thetas.size):
            theta = thetas[i]
            x = r * np.sin(theta)
            z = r * (1 - np.cos(theta))
            distance = [sqrt(
                (x - self.antenna_pos[0, 0]) ** 2 + self.antenna_pos[0, 1] ** 2 + (z - self.antenna_pos[0, 2]) ** 2)]
            if i == 0:
                self.distances = distance
            phase = [4 * np.pi * distance[0] / self.wave_length + np.random.normal(0, 0.01, 1)[0]]
            phases.append(phase)
        return asarray(phases)

    def start_simulation(self):
        simulationData = self.get_simulation_data()[:, 0]
        radiusRange = np.linspace(0.2, 0.6, 10)
        # file = open(unicode('../data/simulationResult.txt', 'utf8'), 'a')
        maxTheta = []
        for oneRadius in radiusRange:
            self.radius = oneRadius
            rk = self.get_rk()
            initialDistance = asarray([self.distances[0]])
            rk.predict_update(initialDistance, self.H_of, self.hx, args=self.antenna_pos, hx_args=self.antenna_pos)
            predicted = []
            for i in range(1, simulationData.size):
                distance = self.get_delta_d(simulationData[i] - simulationData[i - 1])
                initialDistance += asarray([distance])
                rk.predict_update(initialDistance, self.H_of, self.hx, args=self.antenna_pos, hx_args=self.antenna_pos)
                predicted.append(rk.x[0])
                # 计算相位误差
            maxTheta.append(max(predicted))
        plt.figure()
        plt.plot(radiusRange, maxTheta, label=str(self.radius))
        plt.legend()
        plt.show()


def main():
    ekf=RotatingEKF()
    ekf.start_simulation()
    return


if __name__ == "__main__":
    main()
