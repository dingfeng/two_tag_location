# -*- coding: UTF-8 -*-
# filename: Phase_Fsolve date: 2018/2/2 16:15  
# author: FD 
from scipy.optimize import *
from math import *
from scipy.linalg import norm
import numpy as np

d = [[0, 0], [0, 0], [0, 0]]
m = 0


def f(x):
    global m
    global d
    x0 = float(x[0])
    y0 = float(x[1])
    alpha = float(x[2])
    l1 = abs(float(x[3]))
    l2 = abs(float(x[4]))
    l3 = abs(float(x[5]))
    # angle = float(x[6])
    return [
        sqrt(pow(x0 + l1 * cos(alpha), 2) + pow(y0 + l1 * sin(alpha) + m, 2)) - sqrt(pow(x0, 2) + pow(y0 + m, 2)) -
        d[0][0],
        sqrt(pow(x0 + l2 * cos(alpha), 2) + pow(y0 + l2 * sin(alpha) + m, 2)) - sqrt(pow(x0, 2) + pow(y0 + m, 2)) -
        d[1][0],
        sqrt(pow(x0 + l3 * cos(alpha), 2) + pow(y0 + l3 * sin(alpha) + m, 2)) - sqrt(pow(x0, 2) + pow(y0 + m, 2)) -
        d[2][0],

        sqrt(pow(x0 + l1 * cos(alpha), 2) + pow(y0 + l1 * sin(alpha) - m, 2)) - sqrt(pow(x0, 2) + pow(y0 - m, 2)) -
        d[0][1],
        sqrt(pow(x0 + l2 * cos(alpha), 2) + pow(y0 + l2 * sin(alpha) - m, 2)) - sqrt(pow(x0, 2) + pow(y0 - m, 2)) -
        d[1][1],
        sqrt(pow(x0 + l3 * cos(alpha), 2) + pow(y0 + l3 * sin(alpha) - m, 2)) - sqrt(pow(x0, 2) + pow(y0 - m, 2)) -
        d[2][1]
    ]


def init_config(param_m):
    global m
    m = param_m


def get_point(param_d):
    global d
    d = param_d
    results = []
    f_results = []
    for i in range(0,100):
        result = fsolve(f, [1, i, 1, 1, 2, 3], factor=1)
        if abs(result[0]) < 100 and abs(result[0])>0 and abs(result[1]) < 100 and abs(result[1]) > 0:
            results.append([abs(result[0]),abs(result[1])])
            f_results.append(norm(np.array(f(result)), ord=1))
    if results.__len__() > 0:
        sorted_result = np.array(results)[np.argsort(np.array(f_results))]
        sorted_result =  sorted_result[0:min([sorted_result.__len__(), 1])]
        index = np.argmin(f_results)
        min_results = results[index]
        return_result = min_results[0:2]
        return_result[0] = abs(return_result[0])
        return_result[1] = abs(return_result[1])
        return sorted_result
    return [0,0]
