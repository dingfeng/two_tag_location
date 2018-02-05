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

''' 扩展卡尔曼滤波'''


def H_of(x, pos_tags):
    pos_x = float(x[0])
    pos_y = float(x[3])
    shape = pos_tags.shape
    result = np.zeros((shape[0], 6))
    for i in range(shape[0]):
        tag_x = float(pos_tags[i][0])
        tag_y = float(pos_tags[i][1])
        result[i][0] = (pos_x - tag_x) / sqrt((pos_x - tag_x) ** 2 + (pos_y - tag_y) ** 2)
        result[i][3] = (pos_y - tag_y) / sqrt((pos_x - tag_x) ** 2 + (pos_y - tag_y) ** 2)
    return result


def hx(x, pos_tags):
    pos_x = float(x[0])
    pos_y = float(x[3])
    shape = pos_tags.shape
    result = []
    for i in range(shape[0]):
        tag_x = pos_tags[i][0]
        tag_y = pos_tags[i][1]
        result.append(sqrt((pos_x - tag_x) ** 2 + (pos_y - tag_y) ** 2))
    return asarray(result)

rk = None
def get_rk():
    global  rk
    rk = ExtendedKalmanFilter(dim_x=6, dim_z=2)
    dt = 0.01

    rk.F = asarray([[1., dt, dt * dt / 2, 0., 0., 0.],
                    [0., 1., dt, 0., 0., 0.],
                    [0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 1., dt, dt * dt / 2],
                    [0., 0., 0., 0., 1., dt],
                    [0., 0., 0., 0., 0., 1.]])
    # 初始位置 先设置为0
    rk.x = array([0, 0, 1, 0.6, 0, 1]).T
    # 测量误差
    rk.R *= 0.001
    rk.P *= 0.3
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


def get_simulation_data():
    x = np.linspace(0, 0.5, 100)
    y = x ** 2
    plt.figure()
    plt.scatter(x, y)
    phases = []
    antenna_pos = [0, 2]
    for i in range(y.size):
        distance = [sqrt((antenna_pos[0] - (x[i] - 0.05)) ** 2 + (antenna_pos[1] - y[i]) ** 2),
                    sqrt((antenna_pos[0] - (x[i] + 0.05)) ** 2 + (antenna_pos[1] - y[i]) ** 2)]
        phase = [4 * np.pi * distance[0] / wave_length + np.random.normal(0, 0.5, 1)[0],
                 4 * np.pi * distance[1] / wave_length + np.random.normal(0, 0.5, 1)[0]]  # % (2 * np.pi)
        phases.append(phase)
    return asarray(phases)


def start_simulation():
    simulation_data = get_simulation_data() % (2 * np.pi)
    increasement_array = np.zeros((simulation_data.shape[0], simulation_data.shape[1])) + np.pi * 2

    # 增加一定2pi
    for i in range(15):
        rk = get_rk()
        simulation_data = simulation_data + i * increasement_array
        tag0 = simulation_data[:, 0]
        tag1 = simulation_data[:, 1]
        distances = asarray([get_distance_by_phase(tag0[0]), get_distance_by_phase(tag1[0])])
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

    plt.show()
    return


pos_tags = array([[-0.05, 0], [0.05, 0]])
filepath = unicode("../data/active-60cm-40cm-2.csv", "utf8")

def main():

    data = FileReader.read_file(filepath)
    for i in range(15,16):
        rk = get_rk()
        tag0 = FDUtils.interp(data, 1005)+2*np.pi*i
        # tag0 = get_win_data(tag0)
        tag1 = FDUtils.interp(data, 1006)+2*np.pi*i
        # tag1 = get_win_data(tag1)
        # 设置初始相位
        # distances = asarray([get_distance_by_phase(tag0[0]), get_distance_by_phase(tag1[0])])
        distances = asarray([sqrt(0.05**2+0.6**2), sqrt(0.05**2+0.6**2)])
        rk.predict_update(distances, H_of, hx, args=pos_tags, hx_args=pos_tags)
        predicted = []
        for i in range(1, min([tag1.size, tag0.size])):
            tag0_distance_delta = get_delta_d(tag0[i] - tag0[i - 1])
            tag1_distance_delta = get_delta_d(tag1[i] - tag1[i - 1])
            distances += asarray([tag0_distance_delta, tag1_distance_delta])
            rk.predict_update(distances, H_of, hx, args=pos_tags, hx_args=pos_tags)
            predicted.append([rk.x[0], rk.x[3]])
            # print [rk.x[0], rk.x[3]]
        predicted = asarray(predicted)
        predicted1 = np.dot(predicted, asarray([[np.cos(np.pi), -np.sin(np.pi)], [np.sin(np.pi), np.cos(np.pi)]]))
        plt.figure()
        plt.scatter(predicted1[:, 0], predicted1[:, 1])
        plt.figure()
        plt.plot(predicted1[:,0])

    plt.show()
    return


# antenna_x, antenna_y = 0, 1
# pos_tags = array([[-18.5 / 2, 0], [18.5 / 2, 0]])
# predicted = []
# x = np.linspace(-20, 20, int(2000 // dt))
# y = x ** 2
# for i in range(int(2000 // dt)):
#     antenna_x = x[i]
#     antenna_y = y[i]
#     pos_tags_shape = pos_tags.shape
#     measurement_values = []
#     for i in range(pos_tags_shape[0]):
#         tag_x = pos_tags[i][0]
#         tag_y = pos_tags[i][1]
#         measurement_values.append(sqrt((antenna_x - tag_x) ** 2 + (antenna_y - tag_y) ** 2))
#     rk.predict_update(asarray(measurement_values), H_of, hx, args=pos_tags, hx_args=pos_tags)
#     print(rk.x)
#     predicted.append([rk.x[0], rk.x[3]])
#
# predicted = asarray(predicted)
# plt.figure()
# plt.scatter(predicted[:, 0], predicted[:, 1])
# plt.show()



if __name__ == "__main__":
    main()
    # start_simulation()
