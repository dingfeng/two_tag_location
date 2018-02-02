# -*- coding: UTF-8 -*-
# filename: fsolve_method date: 2018/2/2 15:29  
# author: FD 
from dataprecess.FileReader import FileReader
from dataprecess.ImageUtils import ImageUtils
import dataprecess.FDUtils as FDUtils
import numpy as np
import pandas as pd
import scipy as sp
from Phase_Fsolve import *
import matplotlib.pyplot as  plt

filepath = unicode("../data/active_U2.csv", "utf8")

frequency = 920.625e6
wavelength = 3e8 / frequency * 100


def get_window_data(y, win_size=7):
    result = []
    for i in range(y.size / win_size):
        start = i * win_size
        end = start + win_size
        win_mean = np.mean(y[start:end])
        result.append(win_mean)
    return result


def get_distance_delta_by_phase(phase_delta):
    return (phase_delta * wavelength) / (4 * np.pi)


def get_d_delta_by_phase_delta(phase_delta):
    result = []
    for i in range(phase_delta.__len__()):
        result.append([get_distance_delta_by_phase(phase_delta[i][0]), get_distance_delta_by_phase(phase_delta[i][1])])
    return result


# 不进行插值 直接使用读得的相位值 [时间，相位] win_size窗口大小
def get_window_mean_phase(data, win_size=100):
    result = []
    shape = data.shape
    start_timestamp = 0
    in_timestamp_count = 0
    win_total_phase = 0
    for i in range(shape[0]):
        timestamp = data[i, 0]
        phase = data[i, 1]
        if timestamp - start_timestamp < win_size:
            # 在窗口之中
            in_timestamp_count += 1
            win_total_phase += phase
        else:
            # 计算平均值 并加入到列表中
            result.append(win_total_phase / in_timestamp_count)
            # 更新窗口的开始时间戳
            start_timestamp = start_timestamp + win_size
            win_total_phase = phase
            in_timestamp_count = 1
    # 计算最后结尾的时间窗口的平均相位
    if in_timestamp_count > 0:
        mean_phase = win_total_phase / in_timestamp_count
        result.append(mean_phase)
    return result


def main():
    init_config(18.5 / 2)
    data = FileReader.read_file(filepath)
    # 间距为10ms
    first_y = FDUtils.interp(data, 1005)
    ImageUtils.draw_scatter_diagram(range(first_y.size), first_y)
    first__win_y = get_window_data(first_y, 30)
    second_y = FDUtils.interp(data, 1005)
    ImageUtils.draw_scatter_diagram(range(second_y.size), second_y)
    second_win_y = get_window_data(second_y, 30)
    d_delta_list = []
    for i in range(3, min([second_win_y.__len__(), first__win_y.__len__()])):
        phase_deltas = [[first__win_y[i - 2] - first__win_y[i - 3], second_win_y[i - 2] - second_win_y[i - 3]],
                        [first__win_y[i - 1] - first__win_y[i - 3], second_win_y[i - 1] - second_win_y[i - 3]],
                        [first__win_y[i] - first__win_y[i - 3], second_win_y[i] - second_win_y[i - 3]]]
        d_delta = get_d_delta_by_phase_delta(phase_deltas)
        d_delta_list.append(d_delta)
    pos_list = []
    for i in range(d_delta_list.__len__()):
        d_delta = d_delta_list[i]
        points = get_point(d_delta).tolist()
        pos_list += points

    # 画图
    pos_list = np.array(pos_list)
    plt.figure()
    plt.scatter(range(pos_list[:, 0].size), pos_list[:, 0])
    plt.legend()
    plt.figure()
    plt.scatter(range(pos_list[:, 1].size), pos_list[:, 1])
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    main()