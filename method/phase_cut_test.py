# -*- coding: UTF-8 -*-
# filename: phase_cut_test date: 2018/2/2 17:32  
# author: FD 
from  dataprecess.FDUtils import *
from dataprecess.ImageUtils import ImageUtils

# source_file = unicode("../data/V02.csv", "utf8")
dest_file = unicode("../data/active_V02.csv", "utf8")
ImageUtils.draw_phase_diagram(dest_file)
# cut_phase_data(source_file, 3e6, 7e6, dest_file)
