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
import sympy
from sympy import symbols, Matrix
import time
''' 扩展卡尔曼滤波'''


# 求解雅可比行列式模块
class Rotating_Model_Jacobian:
    def __init__(self):
        x, y, z, r, theta, theta_v, theta_a = symbols("x,y,z,r,theta,theta_v,theta_a")
        m = 0.03
        x_tag0 = (r + m) * sympy.cos(theta) + x
        y_tag0 = (r + m) * sympy.sin(theta) + y
        angle_delta = sympy.acos((2 * r ** 2 - m * r) / (2 * r * sympy.sqrt(r ** 2 + m ** 2 - m * r)))
        d = sympy.sqrt(r ** 2 + m ** 2 - m * r)
        x_tag1 = d * sympy.cos(theta - angle_delta) + x
        y_tag1 = d * sympy.sin(theta - angle_delta) + y
        x_tag2 = d * sympy.cos(theta + angle_delta) + x
        y_tag2 = d * sympy.cos(theta + angle_delta) + y
        distance_tag0 = sympy.sqrt(x_tag0 ** 2 + y_tag0 ** 2 + z ** 2)
        distance_tag1 = sympy.sqrt(x_tag1 ** 2 + y_tag1 ** 2 + z ** 2)
        distance_tag2 = sympy.sqrt(x_tag2 ** 2 + y_tag2 ** 2 + z ** 2)
        fxu = Matrix([[distance_tag0], [distance_tag1], [distance_tag2]])
        self.F_J = fxu.jacobian(Matrix([x, y, z,r, theta, theta_v, theta_a]))
        self.x, self.y, self.z, self.r, self.theta, self.theta_v, self.theta_a = x, y, z, r, theta, theta_v, theta_a
        self.subs = {x: 0, y: 0, z: 0, r: 0, theta: 0, theta_v: 0, theta_a: 0}
        pass

    def get_jacobian_array(self, x):
        self.subs[self.x] = x[0]
        self.subs[self.y] = x[1]
        self.subs[self.z] = x[2]
        self.subs[self.r] = x[3]
        self.subs[self.theta] = x[4]
        self.subs[self.theta_v] = x[5]
        self.subs[self.theta_a] = x[6]
        print 'begin  ',time.time()
        F = array(self.F_J.evalf(subs=self.subs)).astype(float)
        print 'end  ',time.time()
        return F


triangle = 0.03
rotating_model_jacobian = None


# pos_tags = np.array(
#     [[0, triangle, 0], [-triangle * sqrt(3) / 2.0, -3.0 / 2, 0], [triangle * sqrt(3) / 2.0, -3.0 / 2.0, 0]])


def H_of(x):
    result = rotating_model_jacobian.get_jacobian_array(x)
    return result


def hx(x):
    x0 = x[0]
    y0 = x[1]
    z0 = x[2]
    r = x[3]
    theta = x[4]
    theta_v = x[5]
    theta_a = x[6]
    x_tag0 = (r + triangle) * cos(theta) + x0
    y_tag0 = (r + triangle) * sin(theta) + y0
    angle_delta = arccos(
        (2 * r ** 2 - triangle * r + triangle ** 2 / 4) / (2 * r * sqrt(r ** 2 + triangle ** 2 - triangle * r)))
    d = sqrt(r ** 2 + triangle ** 2 - triangle * r)
    x_tag1 = d * cos(theta - angle_delta) + x0
    y_tag1 = d * sin(theta - angle_delta) + y0
    x_tag2 = d * cos(theta + angle_delta) + x0
    y_tag2 = d * cos(theta + angle_delta) + y0
    distance_tag0 = sqrt(x_tag0 ** 2 + y_tag0 ** 2 + z0 ** 2)
    distance_tag1 = sqrt(x_tag1 ** 2 + y_tag1 ** 2 + z0 ** 2)
    distance_tag2 = sqrt(x_tag2 ** 2 + y_tag2 ** 2 + z0 ** 2)
    return asarray([distance_tag0, distance_tag1, distance_tag2])


rk = None


def get_rk():
    global rk
    global rotating_model_jacobian
    rotating_model_jacobian = Rotating_Model_Jacobian()
    rk = ExtendedKalmanFilter(dim_x=7, dim_z=3)
    dt = 0.01
    rk.F = asarray([[1., 0., 0., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0., 0., 0.],
                    [0., 0., 1., 0., 0., 0., 0.],
                    [0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 1., dt, dt * dt / 2],
                    [0., 0., 0., 0., 0., 1., dt],
                    [0., 0., 0., 0., 0., 0., 1.]
                    ])
    # 初始位置 先设置为0
    rk.x = array([1, 1, 1, 0.4, 0, 1, 1]).T
    # 测量误差
    rk.R *= 0.001
    rk.P *= 1
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


distances = []


def get_simulation_data():
    global distances
    global triangle
    circle_center = [1, 1, 1]
    r = 0.4
    radians = np.linspace(0, np.pi, 100)
    # x = r * cos(radians) + circle_center[0]
    # y = r * sin(radians) + circle_center[1]
    # z = asarray([circle_center[2] for i in range(x.size)])
    m = triangle
    x_tag0 = (r + m) * cos(radians) + circle_center[0]
    y_tag0 = (r + m) * sin(radians) + circle_center[1]
    z_tag0 = asarray([circle_center[2] for i in range(radians.size)])

    x_tag1 = sqrt(r ** 2 + m ** 2 - m * r) * cos(
        radians - arccos((2 * r ** 2 - m * r) / (2 * r * sqrt(r ** 2 + m ** 2 - m * r)))) + \
             circle_center[0]
    y_tag1 = sqrt(r ** 2 + m ** 2 - m * r) * sin(
        radians - arccos((2 * r ** 2 - m * r) / (2 * r * sqrt(r ** 2 + m ** 2 - m * r)))) + \
             circle_center[1]
    z_tag1 = asarray([circle_center[2] for i in range(radians.size)])

    x_tag2 = sqrt(r ** 2 + m ** 2 - m * r) * cos(
        radians + arccos((2 * r ** 2 - m * r) / (2 * r * sqrt(r ** 2 + m ** 2 - m * r)))) + \
             circle_center[0]
    y_tag2 = sqrt(r ** 2 + m ** 2 - m * r) * sin(
        radians + arccos((2 * r ** 2 - m * r) / (2 * r * sqrt(r ** 2 + m ** 2 - m * r)))) + \
             circle_center[1]
    z_tag2 = asarray([circle_center[2] for i in range(radians.size)])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_tag0, y_tag0, z_tag0)
    phases = []
    for i in range(x_tag0.size):
        distance = [sqrt(x_tag0[i] ** 2 + y_tag0[i] ** 2 + z_tag0[i] ** 2),
                    sqrt(x_tag1[i] ** 2 + y_tag1[i] ** 2 + z_tag1[i] ** 2),
                    sqrt(x_tag2[i] ** 2 + y_tag2[i] ** 2 + z_tag2[i] ** 2)
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
    rk.predict_update(distances, H_of, hx)
    predicted = []
    for i in range(1, min([tag1.size, tag0.size, tag2.size])):
        print i
        tag0_distance_delta = get_delta_d(tag0[i] - tag0[i - 1])
        tag1_distance_delta = get_delta_d(tag1[i] - tag1[i - 1])
        tag2_distance_delta = get_delta_d(tag2[i] - tag2[i - 1])
        distances += asarray([tag0_distance_delta, tag1_distance_delta, tag2_distance_delta])
        rk.predict_update(distances, H_of, hx)
        print rk.x
        predicted.append(rk.x[3])
        # print [rk.x[0], rk.x[3]]
    predicted1 = asarray(predicted)
    # predicted1 = np.dot(predicted, asarray([[np.cos(np.pi), -np.sin(np.pi)], [np.sin(np.pi), np.cos(np.pi)]]))
    # for i in range(predicted1.shape[0]):
    #     print predicted1[i, :]
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(predicted1[:, 0], predicted1[:, 1], predicted1[:, 2])
    plt.figure()
    plt.plot(predicted1)
    plt.show()
    return


if __name__ == "__main__":
    start_simulation()
