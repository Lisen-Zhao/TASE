
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
        tempdata.append(x_data['temp1'].values.tolist()[:])
        return tempdata
    def timegroup3(x_data):
        tempdata = []
        tempdata.append(x_data['temp2'].values.tolist()[:])
        return tempdata
    def timegroup4(x_data):
        tempdata = []
        tempdata.append(x_data['temp3'].values.tolist()[:])
        return tempdata
    tempdata1 = group_data.apply(timegroup2)
    tempdata2 = group_data.apply(timegroup3)
    tempdata3 = group_data.apply(timegroup4)
    load=[]
    temp1=[]
    temp2=[]
    temp3=[]
    for i in range(len(loaddata)):
        onedayload =np.array(loaddata.values.flatten()[i])
        print(onedayload.shape,i)
        onedayload = onedayload.reshape(24,1)
        load.append(onedayload)
    for i in range(len(tempdata1)):
        onedaytemp =np.array(tempdata1.values.flatten()[i])
        onedaytemp = onedaytemp.reshape(24,1)
        temp1.append(onedaytemp)
    for i in range(len(tempdata2)):
        onedaytemp =np.array(tempdata1.values.flatten()[i])
        onedaytemp = onedaytemp.reshape(24,1)
        temp2.append(onedaytemp)
    for i in range(len(tempdata3)):
        onedaytemp =np.array(tempdata1.values.flatten()[i])
        onedaytemp = onedaytemp.reshape(24,1)
        temp3.append(onedaytemp)
    stack_data = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(load)
    stack_data = np.column_stack((np.column_stack((np.column_stack((
        stack_data, TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(temp1))),TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(temp2))),TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(temp3)))
    return stack_data



def data_load():
    data = pd.read_csv('..\\CSVData\\datas.csv', header=0, index_col=0, parse_dates=['date'])

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
        onedayload = onedayload.reshape(24,1)
        load.append(onedayload)
    stack_data = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(load)
    return stack_data