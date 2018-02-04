# -*- coding: UTF-8 -*-
# filename: EKF_method date: 2018/2/4 13:25  
# author: FD 
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import matplotlib.pyplot as plt
import numpy.random as random
from numpy.random import randn
from math import sqrt
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from numpy import array, eye, asarray


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


rk = ExtendedKalmanFilter(dim_x=6, dim_z=2)
dt = 1.
rk.F = array([[1., dt, dt * dt / 2, 0., 0., 0.],
              [0., 1., dt, 0., 0., 0.],
              [0., 0., 1., 0., 0., 0.],
              [0., 0., 0., 1., dt, dt * dt / 2],
              [0., 0., 0., 0., 1., dt],
              [0., 0., 0., 0., 0., 1.]])


def fx(x):
    return np.dot(rk.F, x)


# 初始位置 先设置为0
rk.x = array([1, 1, 1, 1, 1, 1]).T
# 测量误差
rk.R *= 0.01
rk.P *= 0.01
noise_x_var = 0.01
noise_y_var = 0.01
noises = eye(6)
noises[0, 0] = noises[1, 1] = noises[2, 2] = noise_x_var
noises[3, 3] = noises[4, 4] = noises[5, 5] = noise_y_var
rk.Q = np.dot(array([[pow(dt, 4) / 4, pow(dt, 3) / 2, pow(dt, 2) / 2, 0, 0, 0],
                     [pow(dt, 3) / 2, pow(dt, 2), dt, 0, 0, 0],
                     [pow(dt, 2) / 2, dt, 1, 0, 0, 0],
                     [0, 0, 0, pow(dt, 4) / 4, pow(dt, 3) / 2, pow(dt, 2) / 2],
                     [0, 0, 0, pow(dt, 3) / 2, pow(dt, 2), dt],
                     [0, 0, 0, pow(dt, 2) / 2, dt, 1]]), noises)


antenna_x, antenna_y = 0, 1
pos_tags = array([[-1, 0], [1, 0]])
predicted = []
for i in range( int(2000 // dt)):
    antenna_x += 0
    antenna_y += 10
    pos_tags_shape = pos_tags.shape
    measurement_values = []
    for i in range(pos_tags_shape[0]):
        tag_x = pos_tags[i][0]
        tag_y = pos_tags[i][1]
        measurement_values.append(sqrt((antenna_x - tag_x) ** 2 + (antenna_y - tag_y) ** 2))
    rk.predict_update(asarray(measurement_values), H_of, hx, args=pos_tags, hx_args=pos_tags)
    print(rk.x)
    predicted.append([rk.x[0],rk.x[3]])

predicted=asarray(predicted)
plt.figure()
plt.scatter(predicted[:,0],predicted[:,1])
plt.show()