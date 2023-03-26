
import pandas as pd
import numpy as np
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

def data_timeAload():
    data = pd.read_csv('..\\CSVData\\datas.csv', header=0, index_col=0, parse_dates=['date'])

    data_index = data.index.unique()

    group_data = data.groupby(by='date')

    def timegroup(x_data):
        loaddata = []

        loaddata.append(x_data['load'].values.tolist()[:])

        return loaddata
    loaddata = group_data.apply(timegroup)
    def timegroup2(x_data):
        tempdata = []
        tempdata.append(x_data['temp'].values.tolist()[:])
        return tempdata
    tempdata = group_data.apply(timegroup2)
    load=[]
    temp=[]
    for i in range(len(loaddata)):
        onedayload =np.array(loaddata.values.flatten()[i])
        onedayload = onedayload.reshape(48,1)
        load.append(onedayload)
    for i in range(len(tempdata)):
        onedaytemp =np.array(tempdata.values.flatten()[i])
        onedaytemp = onedaytemp.reshape(48,1)
        temp.append(onedaytemp)

    stack_data = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(load)
    stack_data = np.column_stack((stack_data, TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(temp)))
    return stack_data



def data_load():
    data = pd.read_csv('E:\\pythonfile\\Copy_LIu\\CSVData\\Summer.csv', header=0, index_col=0, parse_dates=['date'])

    data_index = data.index.unique()

    group_data = data.groupby(by='date')

    def timegroup(x_data):
        loaddata = []

        loaddata.append(x_data['load'].values.tolist()[:])

        return loaddata

    loaddata = group_data.apply(timegroup)
    load = []
    for i in range(len(loaddata)):
        onedayload =np.array(loaddata.values.flatten()[i])
        onedayload = onedayload.reshape(48,1)
        load.append(onedayload)

    return load