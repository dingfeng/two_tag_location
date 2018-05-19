# -*- coding: UTF-8 -*-
# filename: main date: 2018/4/15 17:03
# author: FD
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt


def cutKinect(data, filepath, startTime, endTime):
    data = data[np.where(data[:, 0] > startTime)[0], :]
    data = data[np.where(data[:, 0] < endTime)[0], :]
    np.savetxt(filepath, data, delimiter=',',comments='',fmt='%f %f %f %f')
    plt.figure()
    plt.scatter(data[:,1],data[:,2])
    plt.show()

def showKinectData(filepath):
    file_content = np.loadtxt(filepath, delimiter=' ')
    x = file_content[:, 0]
    x = x - x[0]
    file_content[:,0] = x
    y = file_content[:, 2]
    plt.figure()
    plt.plot(x, file_content[:, 1], label="X")
    plt.plot(x, y, label='Y')
    plt.plot(x, file_content[:, 3], label="Z")
    plt.legend()
    plt.show()
    return file_content

def main():
    sourcefile = unicode("../data/circular/slow/c4.txt", "utf8")
    destfile = unicode("../data/circular/slow/c4_active.txt", 'utf8')
    print sourcefile
    print destfile
    fileContent = showKinectData(sourcefile)
    startTime = input("start time:")
    endTime = input("end time:")
    cutKinect(fileContent,destfile,startTime,endTime)
    showKinectData(destfile)
    return


if __name__ == '__main__':
    main()
