# -*- coding: UTF-8 -*-
# filename: phase_cut_test date: 2018/2/2 17:32  
# author: FD 
from  dataprecess.FDUtils import *
from dataprecess.ImageUtils import ImageUtils

source_file = unicode("../data/active_three_d_5.csv", "utf8")
dest_file = unicode("../data/active_three_d_5.csv", "utf8")
ImageUtils.draw_phase_diagram(source_file)
# cut_phase_data(source_file, 1.1146e7, 1.855e7, dest_file)
