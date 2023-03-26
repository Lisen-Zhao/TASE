import pandas as pd
import numpy as np
from tslearn.clustering import KShape
import data_process
import matplotlib.pyplot as plt
import math
datacontent = '..\\CSVData\\data_load.csv'
tempcontent = '..\\CSVData\\data_temp.csv'
outputclucsv = '..\\CSVData\\cluster8LTwhole.csv'
num_cluster = 8
seed = 0
stack_data = data_process.data_load(datacontent,tempcontent)
np.random.seed(seed)
data = pd.read_csv(datacontent, header=0, index_col=0, parse_dates=['date'])
data = data.loc['2004-01-01':'2008-06-29'].dropna()
data_index = data.index.unique()
ks = KShape(n_clusters= num_cluster ,n_init= 10 ,verbose= True ,random_state=seed)
y_pred = ks.fit_predict(stack_data)
fin_cluster = pd.DataFrame({"id": data_index, "cluster_label": y_pred})
print(fin_cluster)
fin_cluster.to_csv(outputclucsv)
plt.figure(dpi=150,figsize=(16, 12))
for yi in range(num_cluster):
    plt.subplot(math.ceil(num_cluster / 2), 2, 1 + yi)  #创建小图 2行1列
    for xx in stack_data[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)  #ravel将数组展平成一维
        # print(yi)
        # print(xx.ravel())
    plt.plot(ks.cluster_centers_[yi].ravel(), "r-")
    plt.title("Cluster %d" % (yi + 1))

plt.tight_layout()  #将图像填充整个画布
plt.savefig('..\\figure\\cluster8LTwhole.png')
plt.show()
