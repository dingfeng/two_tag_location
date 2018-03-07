# -*- coding: UTF-8 -*-
# filename: arm_rotate date: 2018/3/1 16:25
# author: FD
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
from numpy import array, eye, asarray, sin, cos, arccos
from dataprecess.FileReader import FileReader
import dataprecess.FDUtils as FDUtils
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm

''' 扩展卡尔曼滤波'''


def H_of(x, pos_tags):
    pos_x = float(x[0])
    pos_y = float(x[3])
    pos_z = float(x[6])
    shape = pos_tags.shape
    result = np.zeros((shape[0], 9))
    for i in range(shape[0]):
        tag_x = float(pos_tags[i][0])
        tag_y = float(pos_tags[i][1])
        tag_z = float(pos_tags[i][2])
        result[i][0] = (pos_x - tag_x) / sqrt((pos_x - tag_x) ** 2 + (pos_y - tag_y) ** 2 + (pos_z - tag_z) ** 2)
        result[i][3] = (pos_y - tag_y) / sqrt((pos_x - tag_x) ** 2 + (pos_y - tag_y) ** 2 + (pos_z - tag_z) ** 2)
        result[i][6] = (pos_z - tag_z) / sqrt((pos_x - tag_x) ** 2 + (pos_y - tag_y) ** 2 + (pos_z - tag_z) ** 2)
    return result


def hx(x, pos_tags):
    pos_x = float(x[0])
    pos_y = float(x[3])
    pos_z = float(x[6])
    shape = pos_tags.shape
    result = []
    for i in range(shape[0]):
        tag_x = pos_tags[i][0]
        tag_y = pos_tags[i][1]
        tag_z = pos_tags[i][2]
        result.append(sqrt((pos_x - tag_x) ** 2 + (pos_y - tag_y) ** 2 + (pos_z - tag_z) ** 2))
    return asarray(result)


rk = None


def get_rk():
    global rk
    rk = ExtendedKalmanFilter(dim_x=9, dim_z=3)
    dt = 0.01

    rk.F = asarray([[1., dt, dt * dt / 2, 0., 0., 0., 0., 0., 0.],
                    [0., 1., dt, 0., 0., 0., 0., 0., 0.],
                    [0., 0., 1., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 1., dt, dt * dt / 2, 0., 0., 0.],
                    [0., 0., 0., 0., 1., dt, 0., 0., 0.],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 1., dt, dt * dt / 2],
                    [0., 0., 0., 0., 0., 0., 0., 1.0, dt],
                    [0., 0., 0., 0., 0., 0., 0., 0., 1.0]])
    # 初始位置 先设置为0
    # rk.x = array([0., 0., 1., 0., -0.473, 1.0, 0.6, 0., 1.]).T
    rk.x = array([0., 0., 1.0, 0., 0, 1.0, 0.9, 0.0, 0.0]).T
    # 测量误差
    rk.R *= 0.001
    rk.P *= 1
    rk.P[6, 6] = 0.0
    rk.P[7, 7] = 0.0
    rk.P[8, 8] = 0.00
    rk.Q *= 0.001
    rk.Q[6, 6] = 0.0
    rk.Q[7, 7] = 0.0
    rk.Q[8, 8] = 0.0
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


tag_r = 0.05
pos_tags = np.array(
    [[0, tag_r, 0], [-tag_r * sqrt(3) / 2.0, -tag_r / 2.0, 0], [tag_r * sqrt(3) / 2.0, -tag_r / 2.0, 0]])

distances = []


def get_simulation_data():
    global distances
    x = np.linspace(0, 0.5, 100)
    y = x ** 2
    z = x ** 2 + 1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    plt.figure()
    plt.scatter(x, y)
    phases = []
    for i in range(y.size):
        distance = [sqrt((x[i] - pos_tags[0, 0]) ** 2 + (y[i] - pos_tags[0, 1]) ** 2 + (z[i] - pos_tags[0, 2]) ** 2),
                    sqrt((x[i] - pos_tags[1, 0]) ** 2 + (y[i] - pos_tags[1, 1]) ** 2 + (z[i] - pos_tags[1, 2]) ** 2),
                    sqrt((x[i] - pos_tags[2, 0]) ** 2 + (y[i] - pos_tags[2, 1]) ** 2 + (z[i] - pos_tags[2, 2]) ** 2)
                    ]
        if i == 0:
            distances = distance
        phase = [4 * np.pi * distance[0] / wave_length + np.random.normal(0, 0.5, 1)[0],
                 4 * np.pi * distance[1] / wave_length + np.random.normal(0, 0.5, 1)[0],
                 4 * np.pi * distance[2] / wave_length + np.random.normal(0, 0.5, 1)[0]]  # % (2 * np.pi)
        phases.append(phase)

    return asarray(phases)


def start_simulation():
    global distances
    simulation_data = get_simulation_data() % (2 * np.pi)
    # 增加一定2pi
    # for i in range(15):
    rk = get_rk()
    tag0 = simulation_data[:, 0]
    tag1 = simulation_data[:, 1]
    tag2 = simulation_data[:, 2]
    rk.predict_update(distances, H_of, hx, args=pos_tags, hx_args=pos_tags)
    predicted = []
    for i in range(1, min([tag1.size, tag0.size, tag2.size])):
        tag0_distance_delta = get_delta_d(tag0[i] - tag0[i - 1])
        tag1_distance_delta = get_delta_d(tag1[i] - tag1[i - 1])
        tag2_distance_delta = get_delta_d(tag2[i] - tag2[i - 1])
        distances += asarray([tag0_distance_delta, tag1_distance_delta, tag2_distance_delta])
        rk.predict_update(distances, H_of, hx, args=pos_tags, hx_args=pos_tags)
        predicted.append([rk.x[0], rk.x[3], rk.x[6]])
        # print [rk.x[0], rk.x[3]]
    predicted1 = asarray(predicted)
    # predicted1 = np.dot(predicted, asarray([[np.cos(np.pi), -np.sin(np.pi)], [np.sin(np.pi), np.cos(np.pi)]]))
    for i in range(predicted1.shape[0]):
        print predicted1[i, :]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(predicted1[:, 0], predicted1[:, 1], predicted1[:, 2])

    plt.show()
    return


def main():
    filepath = unicode("../data/active_three_d_5.csv", "utf8")
    data = FileReader.read_file(filepath)
    tag0 = FDUtils.interp(data, 9006)
    tag1 = FDUtils.interp(data, 9026)
    tag2 = FDUtils.interp(data, 9027)
    tag0_pos = pos_tags[0]
    tag1_pos = pos_tags[1]
    tag2_pos = pos_tags[2]
    antenna_pos = np.array([0, 0.447, 0.9])
    distances = asarray(
        [norm(antenna_pos - tag0_pos, 2), norm(antenna_pos - tag1_pos, 2), norm(antenna_pos - tag2_pos, 2)])
    # distances = asarray([get_delta_d(tag0[0]),get_delta_d(tag1[0]),get_delta_d(tag2[0])])
    rk = get_rk()
    rk.predict_update(distances, H_of, hx, args=pos_tags, hx_args=pos_tags)
    predicted = []
    for i in range(1, min([tag0.size, tag1.size, tag2.size])):
        tag0_distance_delta = get_delta_d(tag0[i] - tag0[i - 1])
        tag1_distance_delta = get_delta_d(tag1[i] - tag1[i - 1])
        tag2_distance_delta = get_delta_d(tag2[i] - tag2[i - 1])
        distances += asarray([tag0_distance_delta, tag1_distance_delta, tag2_distance_delta])
        rk.predict_update(distances, H_of, hx, args=pos_tags, hx_args=pos_tags)
        predicted.append([rk.x[0], rk.x[3], rk.x[6]])
        print rk.x
    predicted = asarray(predicted)
    plt.figure()
    plt.plot(asarray(range(predicted[:, 2].size)), predicted[:, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(predicted[:, 0], predicted[:, 1], predicted[:, 2])
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(predicted[:, 0][100:], predicted[:, 1][100:], predicted[:, 2][100:])
    plt.show()


if __name__ == "__main__":
    # start_simulation()
    main()
