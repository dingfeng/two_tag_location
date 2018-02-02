# -*- coding: UTF-8 -*-
# filename: Probality_Location date: 2018/1/31 13:45  
# author: FD 
import numpy as np
import scipy as sp
import math
from scipy.linalg import norm
import matplotlib.pyplot as plt
unit = 0.01
anttena_pos = np.array([0, 0])
pos = np.array([10, 0])
current_pos = np.array([10,0])

wave_length = 1.0 / 920e6 * 3e8  #920MHz时的波长
# 模拟时间流动，物体运动,生成观测数据
def get_radian(degree):
    return degree / 180.0 * np.pi

#获得一个圆上的分布点的位置矩阵
def get_pos_offset_on_circle(r, num):
    circle_pos = np.zeros((num, 2))
    radian_delta = 2 * np.pi / num
    for i in range(num):
        angle = radian_delta * i
        circle_pos[i,0] = r * np.cos(angle)
        circle_pos[i,1] = r * np.sin(angle)
    return circle_pos

#获得下一步可能的点 概率相同
def  get_around_pos(pos,scale=1,count=2):
    result= [[0,0]]
    for i in range(count):
        r = (i+1) * scale
        num = 8 * np.power(2,i)
        circle_pos = get_pos_offset_on_circle(r,num)
        result += circle_pos.tolist()
    return np.array(result)


def get_tag0(pos):
    return pos + np.array([0, 1])


def get_tag1(pos):
    return pos + np.array([0, -1])

# 发射概率
def get_probablity(pre_pos, pos, phase_delta0, phase_delta1):
    tag0_prepos = get_tag0(pre_pos)
    tag0_prepos_distance = norm(tag0_prepos - anttena_pos, ord=2) * unit
    tag0_pos = get_tag0(pos)
    tag0_pos_distance = norm(tag0_pos - anttena_pos, ord=2) * unit
    theory_tag0_phase_delta = 4 * np.pi * (tag0_pos_distance - tag0_prepos_distance)  / wave_length
    tag1_prepos = get_tag1(pre_pos)
    tag1_prepos_distance = norm(tag1_prepos - anttena_pos, ord=2) * unit
    tag1_pos = get_tag1(pos)
    tag1_pos_distance = norm(tag1_pos - anttena_pos, ord=2) * unit
    theory_tag1_phase_delta = 4 * np.pi * (tag1_pos_distance - tag1_prepos_distance) / wave_length
    p =  (1 - np.abs(theory_tag0_phase_delta - phase_delta0) / (4 * np.pi)) * (
        1 - np.abs(theory_tag1_phase_delta - phase_delta1) / (4 * np.pi) )
    return p



# 假定初始位置 未知 使用HMM估算起始点
def get_phase(anttena_pos, tag_pos):
    distance = norm(anttena_pos - tag_pos, ord=2) * unit
    phase = 4 * np.pi * distance / wave_length
    noise = 0#np.random.normal(0, 0.1)
    noised_phase = phase + noise
    return noised_phase

# 模拟时间流动，物体运动,生成观测数据
def generate_observation_data():
    step = 1
    num = 2
    data = np.empty((2, num))
    for i in range(num):
        pos_delta = np.array([i * step, i * step])
        current_pos  = pos + pos_delta
        data[:, i] = np.array([get_phase(anttena_pos,get_tag0(current_pos)) ,get_phase(anttena_pos,get_tag1(current_pos))])
    return data

def draw_points(points):
    plt.figure()
    plt.title("points graph")
    plt.scatter(points[:,0],points[:,1],marker='o',c='r',label="candidate points")
    plt.legend()
    plt.show()
    return


def main():
    global  pos
    data = generate_observation_data()
    around_poses = get_around_pos(pos)
    p_list=[]
    pre_pos = np.array(pos)
    for i in range(around_poses.shape[0]):
        one_point = around_poses[i,:]
        p = get_probablity(pre_pos,one_point+pre_pos,data[0,1] - data[0,0] , data[1,1] - data[1,0])
        print str(one_point[0])+" "+str(one_point[1])+" "+str(p)
        p_list.append(p)
    candidate_poses = around_poses[(-np.array(p_list)).argsort(),:][range(0,2),:]
    draw_points(candidate_poses)
    #打印信息
    return

if __name__ == "__main__":
    main()