# -*- coding: UTF-8 -*-
# filename: TagIntervals date: 2018/4/28 9:57  
# author: FD 
import numpy as np
import pylab
import scipy.stats as stats
import matplotlib.pyplot as plt
filepath = unicode("../data/lab-2018-4-20/active_v1.csv", "utf8")
csvData = np.loadtxt(filepath, delimiter=',', skiprows=1)
epcs = np.unique(csvData[:, 0])


for epc in epcs:
    epcTimeData = csvData[np.where(csvData[:, 0] == epc)[0], 3]
    rolledEpcTimeData = np.roll(epcTimeData, 1)
    timeIntervals = (epcTimeData - rolledEpcTimeData)[1:]
    minTimeInterval=np.min(timeIntervals)
    maxTimeInterval=np.max(timeIntervals)
    mean=np.mean(timeIntervals)
    median=np.median(timeIntervals)
    std=np.std(timeIntervals)
    sum=epcTimeData[-1]-epcTimeData[0]
    print 'size= ',timeIntervals.size,' minTimeInterval= ',minTimeInterval,' maxTimeInterval= ',maxTimeInterval,' mean= ',mean, ' median= ',median,' sum= ',sum,' std= ',std
    # stats.probplot(timeIntervals, dist="norm", plot=pylab)
    plt.hist(timeIntervals,bins=50,normed=True)
    poissonDataSet = np.random.poisson(1.0/mean, timeIntervals.size)
    print stats.ks_2samp(1.0/timeIntervals, poissonDataSet)
    pylab.show()


