# -*- coding: UTF-8 -*-
# filename: PhaseDistribution date: 2018/4/17 15:37  
# author: FD 
import numpy as np
import numpy as np
import pylab
import scipy.stats as stats

filepath = unicode("../data/active_h1.csv", "utf8")
csvData = np.loadtxt(filepath, delimiter=',', skiprows=1)
epcs = np.unique(csvData[:, 0])
for epc in epcs:
    dataOfEpc = csvData[np.where(csvData[:, 0] == epc)[0], 5]
    mean = np.mean(dataOfEpc)
    std = np.sqrt(np.var(dataOfEpc) * dataOfEpc.size / (dataOfEpc.size - 1))
    print 'epc=',epc,' mean=',mean,' std=',std
    pylab.figure()
    stats.probplot(dataOfEpc, dist="norm", plot=pylab)

pylab.show()
