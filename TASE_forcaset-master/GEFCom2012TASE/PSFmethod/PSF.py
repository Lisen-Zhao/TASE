# import sys
# sys.path.append("E:\pythonfile\GEFCom2012\Clustermethod\Kshapeclusterget.py")
import pandas as pd
import datetime
import numpy as np
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
def PSF_result(data,dataname,ps_index,aim_date):
    aimdata  = pd.read_csv(dataname, header=0, index_col=0, parse_dates=['date'])
    weekdayseason = pd.read_csv('..\\CSVData\\weekdayseason.csv', header=0, index_col=0, parse_dates=['date'])
    allvalue = []
    aimdata_index = aimdata.index.unique()
    sc_load = MinMaxScaler()
    # data_mm = pd.DataFrame(None)
    load=(aimdata.loc[aimdata_index].values.reshape(-1,1)).reshape(-1)

    ilist = np.stack(12 * (aimdata_index, aimdata_index), axis=1).reshape(-1)
    data_mm = pd.DataFrame(load,index=ilist)
    data_mm.columns=['load']
    data_index = data.index.unique()
    allvalue = np.array([])
    clvalue=[]
    finvalue =[]
    num_cluster = 3
    seed = 0
    value1 = []
    value2 = []
    for i in ps_index:
        if np.max(weekdayseason.loc[i, 'weekday']) == np.max(weekdayseason.loc[aim_date, 'weekday']):
            onedayload = np.array(data_mm.loc[i, 'load'].values.flatten())

            onedayload = onedayload.reshape(24, 1)
            value1.append(onedayload)
        else:
            onedayload = np.array(data_mm.loc[i, 'load'].values.flatten())
            onedayload = onedayload.reshape(24, 1)
            value2.append(onedayload)
    print('week value1=', len(value1))
    print('week value2=', len(value2))
    if len(value1) != 0:

        value1 = sum(value1) / len(value1)
        # value2 = sum(value2)/len(value2)
        psfpre = value1 * 1
    # elif len(value1)==0:
    #
    #         psfpre = sum(value1)/len(value1)
    #         print(i,'havent value1')
    else:
        for i in ps_index:
            if np.max(weekdayseason.loc[i, 'season']) == np.max(weekdayseason.loc[aim_date, 'season']):
                onedayload = np.array(data_mm.loc[i, 'load'].values.flatten())

                onedayload = onedayload.reshape(24, 1)
                value1.append(onedayload)
        print('season value1=', len(value1))
        print('season value2=', len(value2))
        if len(value1) != 0:
            psfpre = sum(value1) / len(value1)
            print(aim_date, 'have season value1')
        else:
            # value2=[]
            # value2.append(data_mm.loc[aim_date, 'load'].values.reshape(48,1))
            valuemid = sum(value2) / len(value2)
            value2 = []
            value2.append(valuemid)
            value2.append(data_mm.loc[aim_date, 'load'].values.reshape(24, 1))
            psfpre = sum(value2) / len(value2)
            print(aim_date, 'havent value1')
        # onedayload = np.array(data.loc[i].values.flatten())
        # onedayload = onedayload.reshape(24,1)
        # allvalue=np.append(allvalue ,onedayload)
    # fin_cluster = pd.DataFrame(allvalue.reshape(-1,24), columns=[str(x) + 'h' for x in range(1,25)], index=ps_index)


    # choice_data_index = fin_cluster.apply(lambda x: np.mean(np.abs(x - data.loc[predate].values) / data.loc[predate].values) * 100,
    #                                        axis=1).sort_values()[:1].index
    # for i in choice_data_index:
    #     fin_clusterload = np.array(data.loc[i].values.flatten())
    #     fin_clusterload = fin_clusterload.reshape(24, 1)
    #     finvalue.append(fin_clusterload)
    #     print(i)
    # psfpre = sum(finvalue)/len(finvalue)

    return psfpre


def PSF_dayouttryw6T(csv_name, pred_date, data_index):
    cldata = pd.read_csv(csv_name, header=0, index_col=1, parse_dates=['id'])
    # cldata = cldata.loc['2004-06-01':'2007-08-31'].dropna()
    sday = []
    # pred_date = datetime.datetime.strptime(pred_date, "%Y-%m-%d")
    sday.append(cldata.loc[pred_date,'cluster_label'])
    sday.append(cldata.loc[pred_date+datetime.timedelta(days=-1),'cluster_label'])
    sday.append(cldata.loc[pred_date+datetime.timedelta(days=-2),'cluster_label'])
    sday.append(cldata.loc[pred_date+datetime.timedelta(days=-3),'cluster_label'])
    sday.append(cldata.loc[pred_date+datetime.timedelta(days=-4),'cluster_label'])
    sday.append(cldata.loc[pred_date+datetime.timedelta(days=-5),'cluster_label'])
    ps_index = np.array([])
    for i in data_index:
        try:
            if i+datetime.timedelta(days=6) in data_index:
                if cldata.loc[i,'cluster_label'] == sday[5] and cldata.loc[
                    i+datetime.timedelta(days=1),'cluster_label'] == sday[4] and cldata.loc[
                    i+datetime.timedelta(days=2),'cluster_label'] == sday[3] and cldata.loc[
                    i+datetime.timedelta(days=3),'cluster_label'] == sday[2] and cldata.loc[
                    i+datetime.timedelta(days=4),'cluster_label'] == sday[1] and cldata.loc[
                    i+datetime.timedelta(days=5),'cluster_label'] == sday[0] and \
                        i+datetime.timedelta(days=6) != pred_date+datetime.timedelta(days=1):
                    ps_index = np.append(ps_index, i+datetime.timedelta(days=6))
        except Exception:
            continue
    for i in data_index:
        try:
            if i+datetime.timedelta(days=-5) in data_index and i+datetime.timedelta(days=1) in data_index:
                if cldata.loc[i,'cluster_label'] == sday[0] and cldata.loc[
                    i+datetime.timedelta(days=-1),'cluster_label'] == sday[1] and cldata.loc[
                    i+datetime.timedelta(days=-2),'cluster_label'] == sday[2] and cldata.loc[
                    i+datetime.timedelta(days=-3),'cluster_label'] == sday[3] and cldata.loc[
                    i+datetime.timedelta(days=-4),'cluster_label'] == sday[4] and cldata.loc[
                    i+datetime.timedelta(days=-5),'cluster_label'] == sday[5] and \
                        i+datetime.timedelta(days=1) not in ps_index and \
                        i+datetime.timedelta(days=1) != pred_date+datetime.timedelta(days=1):
                    ps_index = np.append(ps_index, i+datetime.timedelta(days=1))
        except Exception:
            continue
    print('window = 6')
    print('modelen=',len(ps_index))
    if len(ps_index)<=4:
        print('window = 5')

        for i in data_index:
            try:
                if i+datetime.timedelta(days=5) in data_index:
                    if cldata.loc[i,'cluster_label'] == sday[4] and cldata.loc[
                        i+datetime.timedelta(days=1),'cluster_label'] == sday[3] and cldata.loc[
                        i+datetime.timedelta(days=2),'cluster_label'] == sday[2] and cldata.loc[
                        i+datetime.timedelta(days=3),'cluster_label'] == sday[1] and cldata.loc[
                        i+datetime.timedelta(days=4),'cluster_label'] == sday[0] \
                            and i+datetime.timedelta(days=5) != pred_date+datetime.timedelta(days=1):
                        ps_index = np.append(ps_index, i+datetime.timedelta(days=5))
            except Exception:
                continue
        for i in data_index:
            try:
                if i+datetime.timedelta(days=-4) in data_index and i+datetime.timedelta(days=1) in data_index:
                    if cldata.loc[i,'cluster_label'] == sday[0] and cldata.loc[
                        i+datetime.timedelta(days=-1),'cluster_label'] == sday[1] and cldata.loc[
                        i+datetime.timedelta(days=-2),'cluster_label'] == sday[2] and cldata.loc[
                        i+datetime.timedelta(days=-3),'cluster_label'] == sday[3] and cldata.loc[
                        i+datetime.timedelta(days=-4),'cluster_label'] == sday[4] and  \
                            i+datetime.timedelta(days=1) not in ps_index and \
                            i+datetime.timedelta(days=1) != pred_date+datetime.timedelta(days=1):
                        ps_index = np.append(ps_index, i+datetime.timedelta(days=1))
            except Exception:
                continue
        print('modelen=',len(ps_index))
    if len(ps_index)<=4:
        print('window = 4')

        for i in data_index:
            try:
                if i+datetime.timedelta(days=4) in data_index:
                    if cldata.loc[i,'cluster_label'] == sday[3] and cldata.loc[
                        i+datetime.timedelta(days=1),'cluster_label'] == sday[2] and cldata.loc[
                        i+datetime.timedelta(days=2),'cluster_label'] == sday[1] and cldata.loc[
                        i+datetime.timedelta(days=3),'cluster_label'] == sday[0]  \
                            and i+datetime.timedelta(days=4) != pred_date+datetime.timedelta(days=1):
                        ps_index = np.append(ps_index, i+datetime.timedelta(days=4))
            except Exception:
                continue
        for i in data_index:
            try:
                if i+datetime.timedelta(days=-3) in data_index and i+datetime.timedelta(days=1) in data_index:
                    if cldata.loc[i,'cluster_label'] == sday[0] and cldata.loc[
                        i+datetime.timedelta(days=-1),'cluster_label'] == sday[1] and cldata.loc[
                        i+datetime.timedelta(days=-2),'cluster_label'] == sday[2] and cldata.loc[
                        i+datetime.timedelta(days=-3),'cluster_label'] == sday[3] and   \
                            i+datetime.timedelta(days=1) not in ps_index and \
                            i+datetime.timedelta(days=1) != pred_date+datetime.timedelta(days=1):
                        ps_index = np.append(ps_index, i+datetime.timedelta(days=1))
            except Exception:
                continue
        print('modelen=',len(ps_index))
    if len(ps_index) <=4:
        print('window = 3')

        for i in data_index:
            try:
                if i + datetime.timedelta(days=3) in data_index:
                    if cldata.loc[i, 'cluster_label'] == sday[2] and cldata.loc[
                        i + datetime.timedelta(days=1), 'cluster_label'] == sday[1] and cldata.loc[
                        i + datetime.timedelta(days=2), 'cluster_label'] == sday[0] and i + datetime.timedelta(
                            days=3) != pred_date + datetime.timedelta(
                            days=1):  # 分别代表 i当天模式与预测前三日 i+1天与预测前两日 i+2天与预测前一日 i+3天为待预测日
                        ps_index = np.append(ps_index, i + datetime.timedelta(days=3))
            except Exception:
                continue
        for i in data_index:
            try:

                if i + datetime.timedelta(days=-2) in data_index and i + datetime.timedelta(days=1) in data_index:
                    if cldata.loc[i, 'cluster_label'] == sday[0] and cldata.loc[
                        i + datetime.timedelta(days=-1), 'cluster_label'] == sday[1] and cldata.loc[
                        i + datetime.timedelta(days=-2), 'cluster_label'] == sday[2] and i + datetime.timedelta(
                            days=1) not in ps_index and i + datetime.timedelta(days=1) != pred_date + datetime.timedelta(
                            days=1):
                        ps_index = np.append(ps_index, i + datetime.timedelta(days=1))
            except Exception:
                continue
        print('modelen=', len(ps_index))
    if len(ps_index) == 0:
        print('window = 2')
        for i in data_index:
            try:
                if i + datetime.timedelta(days=2) in data_index:
                    if cldata.loc[i, 'cluster_label'] == sday[1] and cldata.loc[
                        i + datetime.timedelta(days=1), 'cluster_label'] == sday[0] and i + datetime.timedelta(
                            days=2) != pred_date + datetime.timedelta(days=1):  # 分别代表 i当天模式与预测前2日 i+1天与预测前1日 i+2天与预测为待预测日
                        ps_index = np.append(ps_index, i + datetime.timedelta(days=2))
            except Exception:
                continue
        for i in data_index:
            try:

                if i + datetime.timedelta(days=-1) in data_index and i + datetime.timedelta(days=1) in data_index:
                    if cldata.loc[i, 'cluster_label'] == sday[0] and cldata.loc[
                        i + datetime.timedelta(days=-1), 'cluster_label'] == sday[1] and i + datetime.timedelta(
                            days=1) not in ps_index and i + datetime.timedelta(days=1) != pred_date + datetime.timedelta(
                            days=1):
                        ps_index = np.append(ps_index, i + datetime.timedelta(days=1))  # i当天与预测前1日 i-1天与预测前2日 i+1天待预测日
            except Exception:
                continue
        print('modelen=', len(ps_index))
    if len(ps_index) == 0:
        print('window = 1')
        for i in data_index:
            if i + datetime.timedelta(days=1) in data_index:
                if cldata.loc[i, 'cluster_label'] == sday[0]  and i + datetime.timedelta(
                        days=1) != pred_date + datetime.timedelta(days=1):  # 分别代表 i当天模式与预测前2日 i+1天与预测前1日 i+2天与预测为待预测日
                    ps_index = np.append(ps_index, i + datetime.timedelta(days=1))
        for i in data_index:
            if i + datetime.timedelta(days=-1) in data_index:
                if cldata.loc[i + datetime.timedelta(days=-1), 'cluster_label'] == sday[0] and i  not in ps_index \
                        and i != pred_date + datetime.timedelta(days=1):
                    ps_index = np.append(ps_index, i)  # i当天与预测前1日 i-1天与预测前2日 i+1天待预测日

        print('modelen=', len(ps_index))
    # data = pd.read_csv(csv_name, header=0, index_col=0, parse_dates=['date'])
    # weekday=pd.DataFrame(None)
    # for i in ps_index:
    #
    #         weekday = weekday.append(data.loc[i,'weekday'].any())


    return ps_index