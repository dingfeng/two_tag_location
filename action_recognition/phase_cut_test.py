# -*- coding: UTF-8 -*-
# filename: phase_cut_test date: 2018/2/2 17:32
# author: FD
from  dataprecess.FDUtils import *
from dataprecess.ImageUtils import ImageUtils

source_file = unicode("../data/lab-2018-4-20/v1.csv", "utf8")
dest_file = unicode("../data/lab-2018-4-20/active_v1.csv", "utf8")
ImageUtils.draw_phase_diagram(dest_file)
plt.show()
cut_phase_data(source_file, 0, 7500000, dest_file)