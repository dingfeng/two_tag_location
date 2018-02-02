# -*- coding: UTF-8 -*-
# filename: FileReader date: 2017/8/2 22:32  
# author: FD 
import numpy as np


class FileReader(object):
    @classmethod
    def read_file(cls, filename):
        #EPC,Antenna,Channel,TimeStamp/us,RSSI/dBm,Phase/rad,Doppler Shift/Hz,Velocity
        file_content = np.loadtxt(filename,delimiter=',',skiprows=1)
        data = file_content[:,[0,3,4,5]]
        return data
# F:/rfid实验室/字体识别/实验数据/2017-8-2
