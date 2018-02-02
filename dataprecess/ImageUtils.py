# -*- coding: UTF-8 -*-
# filename: ImageUtils date: 2017/8/3 10:54  
# author: FD
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from dataprecess.FileReader import FileReader


class ImageUtils(object):
    @classmethod
    def draw_scatter_diagram(cls, x, y):
        pl.plot(x, y, 'or')
        pl.show()

    @classmethod
    def draw_label_scatter_diagram(cls,x,y,line_label):
        plt.figure()
        plt.plot(x, y, label=line_label)
        plt.legend()
    @classmethod
    def draw_phase_diagram(cls, file_path=None, x=np.array([]), y=np.array([])):
        if file_path is not None:
            data = FileReader.read_file(file_path)
            # EPC, TimeStamp/us, Phase/rad
            epcs = np.unique(data[:, 0])
            plt.figure()
            plt.title("phase graph"+file_path)
            plt.xlabel("time(us)")
            plt.ylabel("phase(rad)")
            for epc in epcs:
                indexes=np.where(data[:,0]==epc)
                epcData=data[indexes,:][0]
                x= np.transpose(epcData[:,1])
                y=np.transpose(epcData[:,3])
                plt.plot(x,y,label=str(int(epc)))
            plt.legend()
            # plt.savefig(file_path+"_phase.png")
            plt.show()
        return

    @classmethod
    def show_figure(cls):
        plt.show()
        return
# ImageUtils.draw_phase_diagram(unicode('F:/rfid实验室/字体识别/实验数据/2017-8-2/1.csv','utf-8'));
# ImageUtils.draw_phase_diagram(x=np.array([1000,2000]),y=np.array([1,2]))
