# -*- coding: UTF-8 -*-
# filename: 2 date: 2018/5/13 16:26  
# author: FD
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

array = [[0.93, 0.07], [0, 1.0]]
df_cm = pd.DataFrame(array, ['v','c'],
                  ['v','c'])
#plt.figure(figsize = (10,7))
sn.set(font_scale=0.8)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 8},cbar=False)# font size
fig=plt.gcf()
fig.set_size_inches(7.03187500000010*0.1,7.0318750000001*0.1)
plt.savefig("confusion-matrix.pdf",
            dpi=1000,
            bbox_inches='tight',)