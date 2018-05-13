# -*- coding: UTF-8 -*-
# filename: main date: 2018/4/15 17:03  
# author: FD
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
filename=unicode("../kinect_data/v2.txt","utf8")
file_content = np.loadtxt(filename,delimiter=' ')
x=file_content[:,0]
x=x-x[0]
y=file_content[:,2]
plt.figure()
plt.plot(x, y, label="Y")
plt.plot(x,file_content[:,1],label="X")
plt.plot(x,file_content[:,3],label="Z")
plt.legend()
plt.show()