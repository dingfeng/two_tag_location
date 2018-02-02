# -*- coding: UTF-8 -*-
# filename: Sample date: 2017/10/21 20:56  
# author: FD

import numpy as np
import pywt
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd
from dataprecess.ImageUtils import ImageUtils

'''
打印错误日志
'''


# unwrap phase
def unwrap(data):
    return np.unwrap(data)


def asymmetricKL(P, Q):
    return sum(P * np.log(P / Q))  # calculate the kl divergence between P and Q


def symmetricalKL(P, Q):
    return (asymmetricKL(P, Q) + asymmetricKL(Q, P)) / 2.00


def interp(data, tag_id):
    data = data[np.where(data[:, 0] == tag_id)[0]]
    data = data[data[:, 1].argsort()]
    time_series = data[:, 1]
    min_interval = float("inf")
    max_interval = -1
    interval_list =[]
    for i in range(1, time_series.size):
        interval = time_series[i] - time_series[i - 1]
        interval_list.append(interval)
        min_interval = min([interval, min_interval])
        max_interval = max([interval, max_interval])
    print "average interval = ",np.mean(interval_list)
    print "min interval = ", min_interval, " max interval = ", max_interval
    data[:, 3] = unwrap(data[:, 3])

    x = data[:, 1] / 1000
    y = data[:, 3]
    f_linear = interpolate.interp1d(x, y)
    maxX = x[x.size - 1]
    minX = x[0]
    timeInterval = 5
    x = np.linspace(minX, maxX, (maxX - minX) / timeInterval)
    y = f_linear(x)
    return y


def cut_phase_data(source_filepath, timestamp_start, timestamp_end, dest_filepath):
    file_content = np.loadtxt(source_filepath, delimiter=',', skiprows=1)
    out_data = [row for row in file_content if timestamp_start < row[3] < timestamp_end]
    out_data = np.array(out_data)
    out_data[:, 3] = out_data[:, 3] - min(out_data[:, 3]);
    np.savetxt(dest_filepath, out_data, delimiter=',',
               header='EPC,Antenna,Channel,TimeStamp/us,RSSI/dBm,Phase/rad,Doppler Shift/Hz,Velocity', comments='')
    return


# cut_phase_data(unicode("F:/rfid实验室/健康/实验数据/第一次/2.csv","utf8"),0,0.5e7,unicode("F:/rfid实验室/健康/实验数据/第一次/2walk.csv","utf8"))




'''
coif6小波降噪 
'''


def denoise(data, type="coif6", mode="periodic", level=None):
    coeff = pywt.wavedec(data, type, mode=mode, level=level)
    uthresh = np.sqrt(2 * np.log(len(coeff)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:])
    z = pywt.waverec(coeff, type)
    return z


def autoRelation(data, t0, t1, lag):
    data0 = data[t0:t1]
    data1 = data[t0 + lag:t1 + lag]
    dcov = np.cov(data0, data1)
    p = dcov[0, 1] / np.sqrt(dcov[0, 0] * dcov[1, 1])
    return p


'''
滑动平均滤波
'''


def movingAve(data, wz):
    result = []
    for i in range(wz, data.size):
        result.append(np.mean(data[i - wz:i]))
    return np.array(result)


''' 
if __name__ == '__main__':
    timestamps=np.array([[1,1.5,1.6,3]]).T;
    values=np.array([[1,4,6,7]]).T;
    data=np.hstack((timestamps,values))
  #  print(data)
    sample_data=do_sample(data,0.5)
    print(sample_data)
    pass
'''
from statsmodels.tsa.stattools import adfuller


def test_stationarity(timeseries):
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    # plt.show(block=False)

    # Perform Dickey-Fuller test:
    print
    'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print dfoutput
