import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.cluster import KMeans
from sklearn import metrics
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import silhouette_score
# from sklearn.metrics import silhouette_score
from tslearn.generators import random_walks
import tslearn.metrics as metrics
from mpl_toolkits.mplot3d import Axes3D
# from tslearn.cycc import cdist_normalized_cc, y_shifted_sbd_vec
import datetime
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import data_process

import glob
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.generators import random_walks





stack_data = data_process.data_load()
stack_data = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(stack_data)
seed = 0
np.random.seed(seed)
sil_score = []
for i in range(2,20):
    num_cluster = i

    ks = KShape(n_clusters= num_cluster ,n_init= 10 ,verbose= True ,random_state=seed)
    y_pred = ks.fit_predict(stack_data)
    dists = ks._my_cross_dist(stack_data)
    print(dists)
    np.fill_diagonal(dists,0)
    score = silhouette_score(dists,y_pred,metric='precomputed')
    sil_score.append(score)
    print(stack_data.shape)
    print(y_pred.shape)
    print("silhouette_score: " + str(score))
    for yi in range(num_cluster):
        plt.subplot(math.ceil(num_cluster/2), 2, yi + 1)
        for xx in stack_data[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.3)
        plt.plot(ks.cluster_centers_[yi].ravel(), "r-")
        plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
                transform=plt.gca().transAxes)
        if yi == 1:
            plt.title("SBD" + "  $k$-shape")

    plt.tight_layout()
    plt.show()
plt.plot(range(2,20), sil_score, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('silhouette_score')
plt.savefig(r'E:\pythonfile\Kshape2-20-silscrealrealonlyload.jpg')
plt.show()