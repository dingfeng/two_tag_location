# -*- coding: UTF-8 -*-
# filename: phase_cut_test date: 2018/2/2 17:32  
# author: FD 
from  dataprecess.FDUtils import *
from dataprecess.ImageUtils import ImageUtils

source_file = unicode("../data/h1.csv", "utf8")
dest_file = unicode("../data/active_h1.csv", "utf8")
ImageUtils.draw_phase_diagram(dest_file)
# cut_phase_data(source_file, 0, 4939000, dest_file)
