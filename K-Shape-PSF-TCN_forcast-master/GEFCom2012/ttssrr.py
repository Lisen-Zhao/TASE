import pandas as pd
import datetime
import numpy as np
dataname = 'E:\\pythonfile\\GEFCom2012\\CSVData\\winter.csv'
data = pd.read_csv(dataname, header=0, index_col=0, parse_dates=['date'])

data = data.loc['2004-01-01':'2008-02-28'].dropna()
data_index = data.index.unique()
cludata = pd.read_csv('E:\\pythonfile\\GEFCom2012\\CSVData\\cluster8LTwhole.csv', header=0, index_col=['id'], parse_dates=['id'])
cluindex = cludata.index.unique()
clvalue = []
finvalue=[]
for i in cluindex:
    for j in cluindex:
        if cludata.loc[i, 'cluster_label'] == cludata.loc[j, 'cluster_label'] and j in data_index:
            clvalue.append(j)
            print(j)
    break
print(len(clvalue))
print(clvalue)
for i in clvalue:
    fin_clusterload = np.array(data.loc[i].values.flatten())
    fin_clusterload = fin_clusterload.reshape(24, 1)
    finvalue.append(fin_clusterload)
# print(len(finvalue))
print(finvalue[11])