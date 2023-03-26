import pandas as pd
import numpy as np
from tslearn.clustering import KShape
import matplotlib.pyplot as plt
import math
from tslearn.clustering import silhouette_score
import data_process
outputclucsv = '..\\CSVData\\cluster6LT.csv'
num_cluster = 6
seed = 0
stack_data = data_process.data_timeAload()
np.random.seed(seed)
data = pd.read_csv('..\\CSVData\\datas.csv', header=0, index_col=0, parse_dates=['date'])
data_index = data.index.unique()
ks = KShape(n_clusters= num_cluster ,n_init= 5 ,verbose= True ,random_state=seed)
y_pred = ks.fit_predict(stack_data)
# dists = ks._my_cross_dist(stack_data)
# np.fill_diagonal(dists,0)
# score = silhouette_score(dists,y_pred,metric='precomputed')
# print(stack_data.shape)
# print(y_pred.shape)
# print("silhouette_score: " + str(score))
fin_cluster = pd.DataFrame({"id": data_index, "cluster_label": y_pred})
print(fin_cluster)
fin_cluster.to_csv(outputclucsv)
for yi in range(num_cluster):
    plt.subplot(math.ceil(num_cluster / 2), 2, yi + 1)
    for xx in stack_data[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.3)
    plt.plot(ks.cluster_centers_[yi].ravel(), "r-")
    plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("SBD" + "  $k$-shape")

plt.tight_layout()
plt.show()