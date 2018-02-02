# -*- coding: UTF-8 -*-
# filename: phase_cut_test date: 2018/2/2 17:32  
# author: FD 
from  dataprecess.FDUtils import *
from dataprecess.ImageUtils import ImageUtils

source_file = unicode("../data/U2.csv", "utf8")
dest_file = unicode("../data/active_U2.csv", "utf8")
ImageUtils.draw_phase_diagram(source_file)
cut_phase_data(source_file, 0.925e7, 2.625e7, dest_file)
