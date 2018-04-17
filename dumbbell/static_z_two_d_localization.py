# -*- coding: UTF-8 -*-
# filename: EKF_method date: 2018/2/4 13:25
# author: FD
# from __future__ import (absolute_import, division, print_function,
#                         unicode_literals)
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

static_z = 0.54


def H_of(x, pos_tags):
    global static_z
    pos_x = float(x[0])
    pos_y = float(x[3])
    shape = pos_tags.shape
    result = np.zeros((shape[0], 6))
    for i in range(shape[0]):
        tag_x = float(pos_tags[i][0])
        tag_y = float(pos_tags[i][1])
        result[i][0] = (pos_x - tag_x) / sqrt((pos_x - tag_x) ** 2 + (pos_y - tag_y) ** 2 + static_z ** 2)
        result[i][3] = (pos_y - tag_y) / sqrt((pos_x - tag_x) ** 2 + (pos_y - tag_y) ** 2 + static_z ** 2)
    return result


def hx(x, pos_tags):
    global static_z
    pos_x = float(x[0])
    pos_y = float(x[3])
    shape = pos_tags.shape
    result = []
    for i in range(shape[0]):
        tag_x = pos_tags[i][0]
        tag_y = pos_tags[i][1]
        result.append(sqrt((pos_x - tag_x) ** 2 + (pos_y - tag_y) ** 2 + static_z ** 2))
    return asarray(result)


rk = None


def get_rk():
    global rk
    rk = ExtendedKalmanFilter(dim_x=6, dim_z=2)
    dt = 0.01

    rk.F = asarray([[1., dt, dt * dt / 2, 0., 0., 0.],
                    [0., 1., dt, 0., 0., 0.],
                    [0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 1., dt, dt * dt / 2],
                    [0., 0., 0., 0., 1., dt],
                    [0., 0., 0., 0., 0., 1.]])
    # 初始位置 先设置为0
    rk.x = array([0, 0, 0.01, 0.8, 0, 0.01]).T
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


def get_simulation_data():
    global distances
    global static_z
    y = np.linspace(0, 0.6, 200)
    x = asarray([0 for i in range(200)])
    # x = y

    plt.figure()
    plt.scatter(x, y)
    phases = []
    antenna_pos = [0, 1]
    for i in range(y.size):
        distance = [sqrt((antenna_pos[0] - (x[i] - 0.04625)) ** 2 + (antenna_pos[1] - y[i]) ** 2 + static_z ** 2),
                    sqrt((antenna_pos[0] - (x[i] + 0.04625)) ** 2 + (antenna_pos[1] - y[i]) ** 2 + static_z ** 2)]
        if i == 0:
            distances = distance
        phase = [4 * np.pi * distance[0] / wave_length + np.random.normal(0, 0.2, 1)[0],
                 4 * np.pi * distance[1] / wave_length + np.random.normal(0, 0.2, 1)[0]]  # % (2 * np.pi)
        phases.append(phase)
    return asarray(phases)


def start_simulation():
    global distances
    global pos_tags
    simulation_data = get_simulation_data()
    plt.figure()
    plt.plot(simulation_data[:, 0])
    plt.figure()
    plt.plot(simulation_data[:, 1])
    # 增加一定2pi
    rk = get_rk()
    simulation_data = simulation_data
    tag0 = simulation_data[:, 0]
    tag1 = simulation_data[:, 1]
    # distances = asarray([get_distance_by_phase(tag0[0]), get_distance_by_phase(tag1[0])])
    rk.predict_update(distances, H_of, hx, args=pos_tags, hx_args=pos_tags)
    predicted = []
    for i in range(1, min([tag1.size, tag0.size])):
        tag0_distance_delta = get_delta_d(tag0[i] - tag0[i - 1])
        tag1_distance_delta = get_delta_d(tag1[i] - tag1[i - 1])
        distances += asarray([tag0_distance_delta, tag1_distance_delta])
        rk.predict_update(distances, H_of, hx, args=pos_tags, hx_args=pos_tags)
        predicted.append([rk.x[0], rk.x[3]])
        print rk.x
        # print [rk.x[0], rk.x[3]]
    predicted = asarray(predicted)
    predicted1 = np.dot(predicted, asarray([[np.cos(np.pi), -np.sin(np.pi)], [np.sin(np.pi), np.cos(np.pi)]]))
    for i in range(predicted1.shape[0]):
        print predicted1[i, :]
    plt.figure()
    plt.scatter(predicted1[:, 0], predicted1[:, 1])
    plt.figure()
    plt.plot(predicted1[:, 1])
    plt.show()
    return


pos_tags = array([[-0.0425, 0], [0.0425, 0]])
filepath = unicode("../data/active_V02.csv", "utf8")


def main():
    data = FileReader.read_file(filepath)
    rk = get_rk()
    tag0 = FDUtils.interp(data, 9028)
    ImageUtils.draw_scatter_diagram(range(tag0.size),tag0)
    # tag0 = get_win_datainterp(tag0)
    tag1 = FDUtils.interp(data, 9029)
    ImageUtils.draw_scatter_diagram(range(tag1.size), tag1)
    # tag1 = get_win_data(tag1)
    # 设置初始相位
    # distances = asarray([get_distance_by_phase(tag0[0]), get_distance_by_phase(tag1[0])])
    distances = asarray([sqrt(0.0425 ** 2 + static_z**2+0.8**2), sqrt(0.0425 ** 2 + static_z**2+0.8**2)])
    rk.predict_update(distances, H_of, hx, args=pos_tags, hx_args=pos_tags)
    predicted = []
    for i in range(1, min([tag1.size, tag0.size])):
        tag0_distance_delta = get_delta_d(tag0[i] - tag0[i - 1])
        tag1_distance_delta = get_delta_d(tag1[i] - tag1[i - 1])
        distances += asarray([tag1_distance_delta, tag0_distance_delta])
        rk.predict_update(distances, H_of, hx, args=pos_tags, hx_args=pos_tags)

        predicted.append([rk.x[0], rk.x[3]])
        # print [rk.x[0], rk.x[3]]
    predicted = asarray(predicted)
    predicted1 = np.dot(predicted, asarray([[np.cos(np.pi), -np.sin(np.pi)], [np.sin(np.pi), np.cos(np.pi)]]))
    plt.figure()
    plt.scatter(predicted1[:, 0], predicted1[:, 1])
    plt.figure()
    plt.plot(predicted1[:, 0])
    plt.figure()
    plt.plot(predicted1[:, 1])
    plt.show()
    return


if __name__ == "__main__":
    main()
    # start_simulation()
