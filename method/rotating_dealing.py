# -*- coding: UTF-8 -*-
# filename: rotating_dealing date: 2018/3/5 21:45  
# author: FD
from dataprecess.FileReader import FileReader
import dataprecess.FDUtils as FDUtils
import matplotlib.pyplot as plt
source_file = unicode("../data/rotating_1.csv", "utf8")
data = FileReader.read_file(source_file)
tag0 = FDUtils.interp(data, 9006)
tag1 = FDUtils.interp(data, 9026)
tag2 = FDUtils.interp(data, 9027)
plt.figure()
plt.plot(tag0,label="tag0")
plt.plot(tag1,label="tag1")
plt.plot(tag2,label="tag2")
plt.legend()
plt.show()
