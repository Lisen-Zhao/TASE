import pandas as pd
import numpy as np
from tslearn.clustering import KShape
import data_process
from tslearn.clustering import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# datacontent = 'E:\\pythonfile\\GEFCom2012\\CSVData\\winter.csv'
tempcontent = '..\\CSVData\\data_temp.csv'
datacontent = '..\\CSVData\\data_load.csv'
# datacontent = '..\\CSVData\\data_load.csv'
# outputclucsv = 'E:\\pythonfile\\MIDW\\CSVData\\cluster6LTwinter.csv'
# num_cluster = 6
seed = 0
stack_data = data_process.data_timeAload(datacontent,tempcontent)
np.random.seed(seed)
data = pd.read_csv(datacontent, header=0, index_col=0, parse_dates=['date'])
data = data.loc['2004-01-01':'2008-06-29'].dropna()
data_index = data.index.unique()
distortions = []
sil_score = []
for i in range(2, 19):
    ks = KShape(n_clusters=i, n_init=10, verbose=True, random_state=seed)
    y_pred = ks.fit_predict(stack_data)
    distortions.append(ks.inertia_)
    sil_score.append(silhouette_score(stack_data, y_pred, metric='dtw'))
    print(i)

# plt.plot(range(1,20), distortions, marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('Distortion')
# plt.savefig(r'E:\pythonfile\Kshape1-20.jpg')
# plt.show()

fig = plt.figure(figsize=(13, 10), dpi=100)
plt.subplot(2, 1, 1)
plt.plot(range(2, 19), distortions, 'o-', color='y', label='inertia')
plt.xlabel('K number')
plt.ylabel('Distortions')
plt.legend(loc="upper right")
plt.subplot(2, 1, 2)
plt.plot(range(2, 19), sil_score, '^-', color='c', label='silhouette score')
plt.xlabel('K number')
plt.ylabel('silhouette score')
plt.legend(loc="upper right")
plt.savefig("..\\figure\\distortionswholedata.png")
plt.show()
# plt.xticks(rotation=70)