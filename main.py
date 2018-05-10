# -*- coding: UTF-8 -*-
# filename: main date: 2018/4/24 21:23  
# author: FD
import numpy as np
import os
epcs = None
def changeEPC(source_path,dest_path):
    global epcs
    file_content = np.loadtxt(source_path, delimiter=',', skiprows=1,dtype=str)
    if epcs is None:
        epcs=np.unique(file_content[:,0])
    count=0
    for epc in epcs:
        num = count
        count+=1
        epcIndexes=np.where(file_content[:,0] == epc)[0]
        data = np.asarray([num for i in range(epcIndexes.size)])
        file_content[epcIndexes,0]=data
    np.savetxt(dest_path, file_content, delimiter=',',fmt='%s,%s,%s,%s,%s,%s,%s,%s',
               header='EPC,Antenna,Channel,TimeStamp/us,RSSI/dBm,Phase/rad,Doppler Shift/Hz,Velocity', comments='')

if __name__ =="__main__":
    destDir=unicode('C:/Users/FD/Desktop/newdata/','utf8')
    sourceDir=unicode('C:/Users/FD/Desktop/20180418/20180418/','utf8')
    files = os.listdir(sourceDir)
    for file in files:
        if not 'With' in file:
            dest_filepath=destDir+file
            source_filepath=sourceDir+file
            changeEPC(source_filepath, dest_filepath)


