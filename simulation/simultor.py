# -*- coding: UTF-8 -*-
# filename: simulation date: 2018/1/29 19:42
# author: FD 
import numpy as np
from scipy.linalg import norm
import math
max_move_speed = 3  # 物体最快运动速度 1 m/s
wave_length = 1.0 / 920e6 * 3e8  #920MHz时的波长
anttena_pos = np.array([0, 0])
pos = np.array([0.1, 0])
current_pos = np.array([0.1,0])

def get_tag0(pos):
    return pos + np.array([0, 0.01])


def get_tag1(pos):
    return pos + np.array([0, -0.01])


# 假定初始位置 未知 使用HMM估算起始点
def get_phase(anttena_pos, tag_pos):
    distance = norm(anttena_pos - tag_pos, ord=2)
    phase = 4 * np.pi * distance / wave_length
    noise = 0#np.random.normal(0, 0.1)
    noised_phase = phase + noise
    return noised_phase


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


def move(pos, pos_delta):
    return pos + pos_delta


# 模拟时间流动，物体运动,生成观测数据
def generate_observation_data():
    step = 0.011 * math.sqrt(2)
    num=7
    data = np.empty((2, num))
    for i in range(num):
        pos_delta = np.array([i * step, 0])
        current_pos  = pos + pos_delta
        data[:, i] = np.array([get_phase(anttena_pos,get_tag0(current_pos)) ,get_phase(anttena_pos,get_tag1(current_pos))])
    return data


# 转移概率
def get_transition_probablity(pos0, pos1):
    # max_distance=max_move_speed * 10e-3
    # pos0_pos1_distance =   math.sqrt((pos0[0] - pos1[0]) * (pos0[0] - pos1[0])+(pos0[1] - pos1[1])*(pos0[1] - pos1[1]))
    # if pos0_pos1_distance <= 1 :
    return 1.0 / 36
    # return 0


# 发射概率
def get_emission_probablity(pre_pos, pos, phase_delta0, phase_delta1):
    tag0_prepos = get_tag0(pre_pos)
    tag0_prepos_distance = norm(tag0_prepos - anttena_pos, ord=2)
    tag0_pos = get_tag0(pos)
    tag0_pos_distance = norm(tag0_pos - anttena_pos, ord=2)
    theory_tag0_phase_delta = 4 * np.pi * (tag0_pos_distance - tag0_prepos_distance) / wave_length
    tag1_prepos = get_tag1(pre_pos)
    tag1_prepos_distance = norm(tag1_prepos - anttena_pos, ord=2)
    tag1_pos = get_tag1(pos)
    tag1_pos_distance = norm(tag1_pos - anttena_pos, ord=2)
    theory_tag1_phase_delta = 4 * np.pi * (tag1_pos_distance - tag1_prepos_distance) / wave_length
    p =  (1 - np.abs(theory_tag0_phase_delta - phase_delta0) / (4 * np.pi)) * (
    1 - np.abs(theory_tag1_phase_delta - phase_delta1) / (4 * np.pi))
    return p


# 预测模型使用维特比算法
def viterbi(obs_seq):
    M=9
    N = M * M  # self.A.shape[0]
    T = obs_seq[0].__len__()
    prev = np.zeros((T - 1, N), dtype=int)
    # DP matrix containing max likelihood of state at a given time
    pos_list = []
    V = np.zeros((N, T))
    for i in range(M):
        x = i * 0.01
        for j in range(M):
            y = j  * 0.01
            pos_list.append([x, y])

    V[0, 0] = 1
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            pos_i = pos_list[i]
            pos_i_index = np.array(pos_i) * 100
            pos_j = pos_list[j]
            pos_j_index = np.array(pos_j) * 100
            # A[i, j] =  #get_transition_probablity(pos_i, pos_j)
            if  abs(pos_i_index[0] - pos_j_index[0]) <= 1 and abs(pos_i_index[0] - pos_j_index[1]) <= 1:
                A[i,j] =  1.0 / 9



    # 计算初始概率
    pre_phase = obs_seq[:, 0]
    for t in range(1, T):
        print "t = ",t
        current_phase = obs_seq[:, t]
        phase_delta = current_phase - pre_phase
        pre_phase = current_phase
        for n in range(N):
            # 重新计算发射概率
            # print "n = ",n
            pos_n = pos_list[n]
            B = np.zeros((N, 1))
            for m in range(N):
                pos_m = pos_list[m]
                B[m, 0] = get_emission_probablity(pos_n, pos_m, phase_delta[0], phase_delta[1])
            B = B/np.sum(B)
            seq_probs = V[:, t - 1] * A[:, n] * B[:,0]
            prev[t - 1, n] = np.argmax(seq_probs)
            V[n, t] = np.max(seq_probs)
    print "gg="
    point_num=np.argmax(V[:, V.shape[1] - 1])
    pos=pos_list[point_num]
    print "position= ("+str(pos[0])+","+str(pos[1])+")"
    return V, prev

#主函数
def main():
    data = generate_observation_data()
    V,prev = viterbi(data)

    return

if __name__ == "__main__":
    main()