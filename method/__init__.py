# -*- coding: UTF-8 -*-
# filename: __init__.py date: 2018/2/2 15:30  
# author: FD 

def f(a,b):
    print a,b

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = [1]
Y = [3]
Z = [1]
ax.scatter(X, Y, Z)
plt.show()