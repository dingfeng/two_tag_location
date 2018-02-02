# -*- coding: UTF-8 -*-
# filename: phase_solve_demo date: 2018/2/1 13:30  
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
        # pow(x0 - 10, 2) + pow(y0, 2) -  r * r * abs(sin(angle))
    ]


def generate_d(x0, y0, m, alpha, ls):
    d = []
    tag0_distance = norm([x0, y0 + m], ord=2)
    tag1_distance = norm([x0, y0 - m], ord=2)
    for l in ls:
        l_tag0_distance = norm([x0 + l * cos(alpha), y0 + l * sin(alpha) + m], ord=2) - tag0_distance
        l_tag1_distance = norm([x0 + l * cos(alpha), y0 + l * sin(alpha) - m], ord=2) - tag1_distance
        d.append([l_tag0_distance, l_tag1_distance])
    return d




def main():
    global d
    global m
    x0 = 100.1
    y0 = 70
    m = 10
    alpha = -pi
    ls = [2, 2 * 2, 3 * 2]
    d = generate_d(x0, y0, m, alpha, ls)
    results=[]
    f_results=[]
    for i in range(1000):
       result = fsolve(f, [i, 1, 3, 1, 2, 3],factor=1)
       results.append(result)
       f_results.append(norm(np.array(f(result)),ord=1))

    index = np.argmin(f_results)
    min_f_results = f_results[index]
    min_results = results[index]
    print "index ",index ," min_f_results ", min_f_results, " min_results= ", min_results
    return


if __name__ == "__main__":
    main()
