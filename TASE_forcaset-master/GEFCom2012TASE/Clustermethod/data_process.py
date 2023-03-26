
import pandas as pd
import numpy as np
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

def data_timeAload(datacontent,tempcontent):
    data = pd.read_csv(datacontent, header=0, index_col=['date'], parse_dates=['date'])
    tempdata = pd.read_csv(tempcontent, header=0, index_col=0, parse_dates=['date'])
    data_index = data.index.unique()
    load_train= data.loc['2004-01-01':'2008-06-29'].dropna()

    group_data = load_train.groupby(by='date')

    def timegroup(x_data):
        loaddata = []

        loaddata.append(x_data.iloc[0].values.tolist()[:])

        return loaddata
    loaddata = group_data.apply(timegroup)
    def timegroup2(x_data, tempdata = tempdata):
        tempdata1 = []
        tempdata1.append(tempdata.loc[x_data.index].values.tolist()[:])
        return tempdata1
    tempdata1 = group_data.apply(timegroup2)
    print(tempdata1)
    # def timegroup3(x_data):
    #     tempdata = []
    #     tempdata.append(x_data['FT'].values.tolist()[:])
    #     return tempdata
    # tempdata2 = group_data.apply(timegroup3)
    # def timegroup4(x_data):
    #     tempdata = []
    #     tempdata.append(x_data['IT'].values.tolist()[:])
    #     return tempdata
    # tempdata3 = group_data.apply(timegroup4)
    load=[]
    temp1=[]
    # temp2 = []
    # temp3 = []

    for i in range(len(loaddata)):
        onedayload =np.array(loaddata.values.flatten()[i])
        onedayload = onedayload.reshape(24,1)
        load.append(onedayload)
    for i in range(len(tempdata1)):
        onedaytemp =np.array(tempdata1.values.flatten()[i])
        onedaytemp = onedaytemp.reshape(24,1)
        temp1.append(onedaytemp)
    # for i in range(len(tempdata2)):
    #     onedaytemp =np.array(tempdata2.values.flatten()[i])
    #     onedaytemp = onedaytemp.reshape(24,1)
    #     temp2.append(onedaytemp)
    # for i in range(len(tempdata3)):
    #     onedaytemp =np.array(tempdata3.values.flatten()[i])
    #     onedaytemp = onedaytemp.reshape(24,1)
    #     temp3.append(onedaytemp)
    stack_data = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(load)
    stack_data = np.column_stack((stack_data, TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(temp1)))
    # stack_data = np.column_stack((stack_data, TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(temp2)))
    # stack_data = np.column_stack((stack_data, TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(temp3)))
    return stack_data



def data_load(datacontent,tempcontent):
    data = pd.read_csv(datacontent, header=0, index_col=['date'], parse_dates=['date'])
    tempdata = pd.read_csv(tempcontent, header=0, index_col=0, parse_dates=['date'])
    data_index = data.index.unique()
    load_train = data.loc['2004-01-01':'2008-06-29'].dropna()

    group_data = load_train.groupby(by='date')

    def timegroup(x_data):
        loaddata = []

        loaddata.append(x_data.iloc[0].values.tolist()[:])

        return loaddata

    loaddata = group_data.apply(timegroup)

    def timegroup2(x_data, tempdata=tempdata):
        tempdata1 = []
        tempdata1.append(tempdata.loc[x_data.index].values.tolist()[:])
        return tempdata1

    tempdata1 = group_data.apply(timegroup2)
    print(tempdata1)
    # def timegroup3(x_data):
    #     tempdata = []
    #     tempdata.append(x_data['FT'].values.tolist()[:])
    #     return tempdata
    # tempdata2 = group_data.apply(timegroup3)
    # def timegroup4(x_data):
    #     tempdata = []
    #     tempdata.append(x_data['IT'].values.tolist()[:])
    #     return tempdata
    # tempdata3 = group_data.apply(timegroup4)
    load = []
    temp1 = []
    # temp2 = []
    # temp3 = []

    for i in range(len(loaddata)):
        onedayload = np.array(loaddata.values.flatten()[i])
        onedayload = onedayload.reshape(24, 1)
        load.append(onedayload)
    # for i in range(len(tempdata1)):
    #     onedaytemp = np.array(tempdata1.values.flatten()[i])
    #     onedaytemp = onedaytemp.reshape(24, 1)
    #     temp1.append(onedaytemp)
    # for i in range(len(tempdata2)):
    #     onedaytemp =np.array(tempdata2.values.flatten()[i])
    #     onedaytemp = onedaytemp.reshape(24,1)
    #     temp2.append(onedaytemp)
    # for i in range(len(tempdata3)):
    #     onedaytemp =np.array(tempdata3.values.flatten()[i])
    #     onedaytemp = onedaytemp.reshape(24,1)
    #     temp3.append(onedaytemp)
    stack_data = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(load)
    # stack_data = np.column_stack((stack_data, TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(temp1)))
    # stack_data = np.column_stack((stack_data, TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(temp2)))
    # stack_data = np.column_stack((stack_data, TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(temp3)))
    return stack_data