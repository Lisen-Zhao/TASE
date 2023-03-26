import pandas as pd
import numpy as np
from tslearn.clustering import KShape

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
fin_cluster = pd.DataFrame({"id": data_index, "cluster_label": y_pred})
print(fin_cluster)
fin_cluster.to_csv(outputclucsv)
