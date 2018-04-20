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

static_z = 0
static_x = 0.0425
static_y = 0.8

def H_of(x, antenna_pos):
    global static_z
    global static_x
    global static_y
    pos_z = float(x[0])
    shape = antenna_pos.shape
    result = np.zeros((shape[0], 3))
    result[0][0] = (pos_z - antenna_pos[0, 2]) / sqrt(
        (static_x - antenna_pos[0, 0]) ** 2 + (static_y - antenna_pos[0, 1]) ** 2 + (pos_z - antenna_pos[0, 2]) ** 2)
    return result


def hx(x, antenna_pos):
    global static_z
    pos_z = float(x[0])
    result = []
    result.append(sqrt(
        (static_x - antenna_pos[0, 0]) ** 2 + (static_y - antenna_pos[0, 1]) ** 2 + (pos_z - antenna_pos[0, 2]) ** 2))
    return asarray(result)


rk = None


def get_rk():
    global rk
    rk = ExtendedKalmanFilter(dim_x=3, dim_z=1)
    dt = 0.01
    rk.F = asarray([[1., dt, dt * dt / 2],
                    [0., 1., dt],
                    [0., 0., 1.],
                    ])
    # 初始位置 先设置为0
    rk.x = array([0, 0, 0]).T
    # 测量误差
    rk.R *= 0.001
    rk.P *= 0.1
    rk.Q *= 0.001
    return rk


def fx(x):
    return np.dot(rk.F, x)


'''扩展卡尔曼滤波'''

'''相位部分'''
frequency = 920.625e6
wave_length = float(3e8 / frequency)


# 单位为cm
def get_distance_by_phase(phase):
    distance = wave_length * phase / (4 * np.pi)
    return distance


# 根据相位差获得距离差
def get_delta_d(delta_phase):
    delta_d = None
    if np.abs(delta_phase) < np.pi:
        delta_d = delta_phase * wave_length / (4 * np.pi)
    elif delta_phase >= np.pi:
        delta_d = (delta_phase - 2 * np.pi) * wave_length / (4 * np.pi)
    elif delta_phase < -np.pi:
        delta_d = (delta_phase + 2 * np.pi) * wave_length / (4 * np.pi)
    return delta_d


distances = None

antenna_pos = asarray([[0, 0, 0.54]])
filepath = unicode("../data/active_V02.csv", "utf8")


def main():
    global antenna_pos
    data = FileReader.read_file(filepath)
    rk = get_rk()
    # tag0 = FDUtils.interp(data, 9028)
    # ImageUtils.draw_scatter_diagram(range(tag0.size), tag0)
    # tag0 = get_win_datainterp(tag0)
    tag1 = FDUtils.interp(data, 9028)
    ImageUtils.draw_scatter_diagram(range(tag1.size), tag1)
    distances = asarray([sqrt(0.0425 ** 2 + 0.54 ** 2 + 0.8 ** 2)])
    rk.predict_update(distances, H_of, hx, args=antenna_pos, hx_args=antenna_pos)
    predicted = []
    for i in range(1, tag1.size):
        tag1_distance_delta = get_delta_d(tag1[i] - tag1[i - 1])
        distances += asarray([tag1_distance_delta])
        rk.predict_update(distances, H_of, hx, args=antenna_pos, hx_args=antenna_pos)
        predicted.append([rk.x[0], rk.x[1], rk.x[2]])
        # print [rk.x[0], rk.x[3]]
    predicted = asarray(predicted)
    # predicted1 = np.dot(predicted, asarray([[np.cos(np.pi), -np.sin(np.pi)], [np.sin(np.pi), np.cos(np.pi)]]))
    plt.figure()
    # plt.scatter(predicted1[:, 0], predicted1[:, 1])
    # plt.figure()
    plt.plot(predicted[:, 0], label="z")
    plt.plot(predicted[:, 1], label="v")
    plt.plot(predicted[:, 2], label="a")
    plt.legend()
    # plt.figure()
    # plt.plot(predicted1[:, 1])
    plt.show()
    return


if __name__ == "__main__":
    main()
    # start_simulation()
