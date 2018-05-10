# -*- coding: UTF-8 -*-
# filename: PhaseDistribution date: 2018/4/17 15:37  
# author: FD
import  matplotlib
from matplotlib import rc
import numpy as np
import pylab
import scipy.stats as stats
rc('text', usetex=True)
filepath = unicode("../data/lab-2018-4-20/v1.csv", "utf8")
csvData = np.loadtxt(filepath, delimiter=',', skiprows=1)
RSSIData=csvData[:,4]
print 'min = ',np.min(RSSIData)
# epcs = np.unique(csvData[:, 0])
# for epc in epcs:
#     dataOfEpc = csvData[np.where(csvData[:, 0] == epc)[0], 4]
#     mean = np.mean(dataOfEpc)
#     std = np.sqrt(np.var(dataOfEpc) * dataOfEpc.size / (dataOfEpc.size - 1))
#     print 'epc=',epc,' mean=',mean,' std=',std
#     stats.probplot(dataOfEpc, dist="norm", plot=pylab)
#     pylab.title("")
#     # pylab.xticks(fontsize=30)
#     # pylab.yticks(fontsize=30)
#     # pylab.ylabel('Ordered Values',fontsize=30,weight='bold')
#     # pylab.xlabel('Theorectical Quantiles',fontsize=30,weight='bold')
# fig=matplotlib.pyplot.gcf()
# fig.set_size_inches(3.3492706944445/2,3.3492706944445/2)
# pylab.savefig("RSSI-qq-plot.pdf",
#             dpi=1000,
#             bbox_inches='tight',)
