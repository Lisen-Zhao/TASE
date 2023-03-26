import pandas as pd
import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def PSF_dayout(csv_name, pred_date, data_index):
    cldata = pd.read_csv(csv_name, header=0, index_col=1, parse_dates=['id'])
    sday = []
    pred_date = datetime.datetime.strptime(pred_date, "%Y-%m-%d")
    sday.append(cldata.loc[pred_date,'cluster_label'])
    sday.append(cldata.loc[pred_date+datetime.timedelta(days=-1),'cluster_label'])
    sday.append(cldata.loc[pred_date+datetime.timedelta(days=-2),'cluster_label'])
    sday.append(cldata.loc[pred_date+datetime.timedelta(days=-3),'cluster_label'])
    sday.append(cldata.loc[pred_date+datetime.timedelta(days=-4),'cluster_label'])
    sday.append(cldata.loc[pred_date+datetime.timedelta(days=-5),'cluster_label'])
    ps_index = np.array([])
    for i in data_index:
        if i+datetime.timedelta(days=6) in data_index:
            if cldata.loc[i,'cluster_label'] == sday[5] and cldata.loc[
                i+datetime.timedelta(days=1),'cluster_label'] == sday[4] and cldata.loc[
                i+datetime.timedelta(days=2),'cluster_label'] == sday[3] and cldata.loc[
                i+datetime.timedelta(days=3),'cluster_label'] == sday[2] and cldata.loc[
                i+datetime.timedelta(days=4),'cluster_label'] == sday[1] and cldata.loc[
                i+datetime.timedelta(days=5),'cluster_label'] == sday[0] and \
                    i+datetime.timedelta(days=6) != pred_date+datetime.timedelta(days=1):
                ps_index = np.append(ps_index, i+datetime.timedelta(days=6))
    for i in data_index:
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

    print('window = 6')
    print('modelen=',len(ps_index))
    if len(ps_index)<=4:
        print('window = 5')

        for i in data_index:
            if i+datetime.timedelta(days=5) in data_index:
                if cldata.loc[i,'cluster_label'] == sday[4] and cldata.loc[
                    i+datetime.timedelta(days=1),'cluster_label'] == sday[3] and cldata.loc[
                    i+datetime.timedelta(days=2),'cluster_label'] == sday[2] and cldata.loc[
                    i+datetime.timedelta(days=3),'cluster_label'] == sday[1] and cldata.loc[
                    i+datetime.timedelta(days=4),'cluster_label'] == sday[0] \
                        and i+datetime.timedelta(days=5) != pred_date+datetime.timedelta(days=1):
                    ps_index = np.append(ps_index, i+datetime.timedelta(days=5))
        for i in data_index:
            if i+datetime.timedelta(days=-4) in data_index and i+datetime.timedelta(days=1) in data_index:
                if cldata.loc[i,'cluster_label'] == sday[0] and cldata.loc[
                    i+datetime.timedelta(days=-1),'cluster_label'] == sday[1] and cldata.loc[
                    i+datetime.timedelta(days=-2),'cluster_label'] == sday[2] and cldata.loc[
                    i+datetime.timedelta(days=-3),'cluster_label'] == sday[3] and cldata.loc[
                    i+datetime.timedelta(days=-4),'cluster_label'] == sday[4] and  \
                        i+datetime.timedelta(days=1) not in ps_index and \
                        i+datetime.timedelta(days=1) != pred_date+datetime.timedelta(days=1):
                    ps_index = np.append(ps_index, i+datetime.timedelta(days=1))

        print('modelen=',len(ps_index))
    if len(ps_index)<=4:
        print('window = 4')

        for i in data_index:
            if i+datetime.timedelta(days=4) in data_index:
                if cldata.loc[i,'cluster_label'] == sday[3] and cldata.loc[
                    i+datetime.timedelta(days=1),'cluster_label'] == sday[2] and cldata.loc[
                    i+datetime.timedelta(days=2),'cluster_label'] == sday[1] and cldata.loc[
                    i+datetime.timedelta(days=3),'cluster_label'] == sday[0]  \
                        and i+datetime.timedelta(days=4) != pred_date+datetime.timedelta(days=1):
                    ps_index = np.append(ps_index, i+datetime.timedelta(days=4))
        for i in data_index:
            if i+datetime.timedelta(days=-3) in data_index and i+datetime.timedelta(days=1) in data_index:
                if cldata.loc[i,'cluster_label'] == sday[0] and cldata.loc[
                    i+datetime.timedelta(days=-1),'cluster_label'] == sday[1] and cldata.loc[
                    i+datetime.timedelta(days=-2),'cluster_label'] == sday[2] and cldata.loc[
                    i+datetime.timedelta(days=-3),'cluster_label'] == sday[3] and   \
                        i+datetime.timedelta(days=1) not in ps_index and \
                        i+datetime.timedelta(days=1) != pred_date+datetime.timedelta(days=1):
                    ps_index = np.append(ps_index, i+datetime.timedelta(days=1))

        print('modelen=',len(ps_index))
    if len(ps_index) ==0:
        print('window = 3')

        for i in data_index:
            if i + datetime.timedelta(days=3) in data_index:
                if cldata.loc[i, 'cluster_label'] == sday[2] and cldata.loc[
                    i + datetime.timedelta(days=1), 'cluster_label'] == sday[1] and cldata.loc[
                    i + datetime.timedelta(days=2), 'cluster_label'] == sday[0] and i + datetime.timedelta(
                        days=3) != pred_date + datetime.timedelta(
                        days=1):  # 分别代表 i当天模式与预测前三日 i+1天与预测前两日 i+2天与预测前一日 i+3天为待预测日
                    ps_index = np.append(ps_index, i + datetime.timedelta(days=3))
        for i in data_index:
            if i + datetime.timedelta(days=-2) in data_index and i + datetime.timedelta(days=1) in data_index:
                if cldata.loc[i, 'cluster_label'] == sday[0] and cldata.loc[
                    i + datetime.timedelta(days=-1), 'cluster_label'] == sday[1] and cldata.loc[
                    i + datetime.timedelta(days=-2), 'cluster_label'] == sday[2] and i + datetime.timedelta(
                        days=1) not in ps_index and i + datetime.timedelta(days=1) != pred_date + datetime.timedelta(
                        days=1):
                    ps_index = np.append(ps_index, i + datetime.timedelta(days=1))

        print('modelen=', len(ps_index))
    if len(ps_index) == 0:
        print('window = 2')
        for i in data_index:
            if i + datetime.timedelta(days=2) in data_index:
                if cldata.loc[i, 'cluster_label'] == sday[1] and cldata.loc[
                    i + datetime.timedelta(days=1), 'cluster_label'] == sday[0] and i + datetime.timedelta(
                        days=2) != pred_date + datetime.timedelta(days=1):  # 分别代表 i当天模式与预测前2日 i+1天与预测前1日 i+2天与预测为待预测日
                    ps_index = np.append(ps_index, i + datetime.timedelta(days=2))
        for i in data_index:
            if i + datetime.timedelta(days=-1) in data_index and i + datetime.timedelta(days=1) in data_index:
                if cldata.loc[i, 'cluster_label'] == sday[0] and cldata.loc[
                    i + datetime.timedelta(days=-1), 'cluster_label'] == sday[1] and i + datetime.timedelta(
                        days=1) not in ps_index and i + datetime.timedelta(days=1) != pred_date + datetime.timedelta(
                        days=1):
                    ps_index = np.append(ps_index, i + datetime.timedelta(days=1))  # i当天与预测前1日 i-1天与预测前2日 i+1天待预测日

        print('modelen=', len(ps_index))
    # if len(ps_index) == 0:
    #     print('window = 1')
    #     for i in data_index:
    #         if i + datetime.timedelta(days=1) in data_index:
    #             if cldata.loc[i, 'cluster_label'] == sday[0]  and i + datetime.timedelta(
    #                     days=1) != pred_date + datetime.timedelta(days=1):  # 分别代表 i当天模式与预测前2日 i+1天与预测前1日 i+2天与预测为待预测日
    #                 ps_index = np.append(ps_index, i + datetime.timedelta(days=1))
    #     for i in data_index:
    #         if i + datetime.timedelta(days=-1) in data_index:
    #             if cldata.loc[i + datetime.timedelta(days=-1), 'cluster_label'] == sday[0] and i  not in ps_index \
    #                     and i != pred_date + datetime.timedelta(days=1):
    #                 ps_index = np.append(ps_index, i)  # i当天与预测前1日 i-1天与预测前2日 i+1天待预测日
    #
    #     print('modelen=', len(ps_index))
    return ps_index



def PSF_result(data,dataname,aim_date,ps_index):
    aimdata  = pd.read_csv(dataname, header=0, index_col=0, parse_dates=['date'])
    allvalue = []
    aimdata_index = aimdata.index.unique()
    sc_load = MinMaxScaler()
    # data_mm = pd.DataFrame(None)
    load=(sc_load.fit_transform(aimdata.loc[:,'load'].values.reshape(-1,1))).reshape(-1)

    ilist = np.stack(24 * (aimdata_index, aimdata_index), axis=1).reshape(-1)
    data_mm = pd.DataFrame(load,index=ilist)
    data_mm.columns=['load']
    # data_mm = data_mm.append(sc_load.fit_transform(data['load'].values.reshape(-1, 1)))
    # for i in ps_index:
        # onedayload = np.array(data_mm.loc[i, 'load'].values.flatten())
        # onedayload = onedayload.reshape(48,1)
        # allvalue.append(onedayload)
    value1 = []
    value2 = []
    for i in ps_index:
        # if np.max(aimdata.loc[i,'weekday']) == np.max(aimdata.loc[aim_date,'weekday']) and np.max(aimdata.loc[aim_date,'weekday'])==0:
        #     onedayload = np.array(data_mm.loc[i,'load'].values.flatten())
        #
        #     onedayload = onedayload.reshape(48,1)
        #     value1.append(onedayload)
        # elif np.max(aimdata.loc[i,'weekday']) == np.max(aimdata.loc[aim_date,'weekday']) and np.max(aimdata.loc[aim_date,'weekday'])==1:
        #     onedayload = np.array(data_mm.loc[i, 'load'].values.flatten())
        #     onedayload = onedayload.reshape(48, 1)
        #     value1.append(onedayload)
        # elif np.max(aimdata.loc[i,'weekday']) ==0:
        #     onedayload = np.array(data_mm.loc[i, 'load'].values.flatten())
        #     onedayload = onedayload.reshape(48, 1)
        #     value2.append(onedayload)
        # else:
        #     onedayload = np.array(data_mm.loc[i, 'load'].values.flatten())
        #     onedayload = onedayload.reshape(48, 1)
        #     value2.append(onedayload)
        if np.max(aimdata.loc[i, 'weekday']) == np.max(aimdata.loc[aim_date, 'weekday']):
            onedayload = np.array(data_mm.loc[i, 'load'].values.flatten())

            onedayload = onedayload.reshape(48, 1)
            value1.append(onedayload)
        # elif np.max(aimdata.loc[i, 'weekday']) == np.max(aimdata.loc[aim_date, 'weekday']) and np.max(
        #         aimdata.loc[aim_date, 'weekday']) == 1:
        #     onedayload = np.array(data_mm.loc[i, 'load'].values.flatten())
        #     onedayload = onedayload.reshape(48, 1)
        #     value1.append(onedayload)
        # elif np.max(aimdata.loc[i, 'weekday']) == 0:
        #     onedayload = np.array(data_mm.loc[i, 'load'].values.flatten())
        #     onedayload = onedayload.reshape(48, 1)
        #     value2.append(onedayload)
        else:
            onedayload = np.array(data_mm.loc[i, 'load'].values.flatten())
            onedayload = onedayload.reshape(48, 1)
            value2.append(onedayload)
    print('week value1=',len(value1))
    print('week value2=', len(value2))
    if len(value1)!=0:

        value1 = sum(value1)/len(value1)
        # value2 = sum(value2)/len(value2)
        psfpre = value1*1
    # elif len(value1)==0:
    #
    #         psfpre = sum(value1)/len(value1)
    #         print(i,'havent value1')
    else:
        for i in ps_index:
            if np.max(aimdata.loc[i, 'season']) == np.max(aimdata.loc[aim_date, 'season']):
                onedayload = np.array(data_mm.loc[i, 'load'].values.flatten())

                onedayload = onedayload.reshape(48, 1)
                value1.append(onedayload)
        print('season value1=', len(value1))
        print('season value2=', len(value2))
        if len(value1)!=0:
            psfpre = sum(value1) / len(value1)
            print(aim_date, 'have season value1')
        else:
            # value2=[]
            # value2.append(data_mm.loc[aim_date, 'load'].values.reshape(48,1))
            valuemid = sum(value2) / len(value2)
            value2 = []
            value2.append(valuemid)
            value2.append(data_mm.loc[aim_date, 'load'].values.reshape(48, 1))
            psfpre = sum(value2) / len(value2)
            print(aim_date, 'havent value1')
    # else:
    #         psfpre = sum(value1) / len(value1)
    #         print(i, 'havent value2')
    # for i in ps_index:
    #     if aimdata.loc[i,'weekday'].any() == aimdata.loc[aim_date,'weekday'].any() and aimdata.loc[aim_date,'weekday'].any()==2: #2和4必须摄取1和3的元素
    #         onedayload = np.array(data.loc[i,'load'].values.flatten())
    #         onedayload = onedayload.reshape(48,1)
    #         allvalue.append(onedayload*0.9)
    #     elif aimdata.loc[i,'weekday'].any() == aimdata.loc[aim_date,'weekday'].any() and aimdata.loc[aim_date,'weekday'].any()==4:
    #         onedayload = np.array(data.loc[i, 'load'].values.flatten())
    #         onedayload = onedayload.reshape(48, 1)
    #         allvalue.append(onedayload * 0.9)
    #     elif aimdata.loc[i,'weekday'].any() == aimdata.loc[aim_date,'weekday'].any() and aimdata.loc[aim_date,'weekday'].any()==1: #1与2对应
    #         onedayload = np.array(data.loc[i, 'load'].values.flatten())
    #         onedayload = onedayload.reshape(48, 1)
    #         allvalue.append(onedayload * 0.6)
    #     elif aimdata.loc[i,'weekday'].any() == aimdata.loc[aim_date,'weekday'].any() and aimdata.loc[aim_date,'weekday'].any()==3: #3与4对应
    #         onedayload = np.array(data.loc[i, 'load'].values.flatten())
    #         onedayload = onedayload.reshape(48, 1)
    #         allvalue.append(onedayload * 0.6)
    #     elif aimdata.loc[i,'weekday'].any() ==1 or aimdata.loc[i,'weekday'].any() ==2:
    #         onedayload = np.array(data.loc[i, 'load'].values.flatten())
    #         onedayload = onedayload.reshape(48, 1)
    #         allvalue.append(onedayload * 0.1)
    #     else:
    #         onedayload = np.array(data.loc[i, 'load'].values.flatten())
    #         onedayload = onedayload.reshape(48, 1)
    #         allvalue.append(onedayload * 1.2)
    # for i in ps_index:
    #     # print(np.max(aimdata.loc[aim_date,'weekday']))  #.all 和 .any .bool 都是对里面所有数是否有ture 输出为ture
    #     if np.max(aimdata.loc[aim_date,'weekday'])==2:
    #         if np.max(aimdata.loc[i,'weekday']) ==1 or np.max(aimdata.loc[i,'weekday'])==2:
    #             onedayload = np.array(data_mm.loc[i, 'load'].values.flatten())
    #             onedayload = onedayload.reshape(48, 1)
    #             # print((onedayload * 0.45).shape,1)
    #             allvalue.append(onedayload * 1.2)
    #         else:
    #             onedayload = np.array(data_mm.loc[i, 'load'].values.flatten())
    #             onedayload = onedayload.reshape(48, 1)
    #             # print((onedayload * 0.45).shape,2)
    #             allvalue.append(onedayload * 0.6)
    #     if np.max(aimdata.loc[aim_date,'weekday'])==4:
    #         if np.max(aimdata.loc[i,'weekday']) ==3 or np.max(aimdata.loc[i,'weekday']) ==4:
    #             onedayload = np.array(data_mm.loc[i, 'load'].values.flatten())
    #             onedayload = onedayload.reshape(48, 1)
    #             # print((onedayload * 0.45).shape, 3)
    #             allvalue.append(onedayload * 1.2)
    #         else:
    #             onedayload = np.array(data_mm.loc[i, 'load'].values.flatten())
    #             onedayload = onedayload.reshape(48, 1)
    #             # print((onedayload * 0.45).shape, 4)
    #             allvalue.append(onedayload * 0.6)
    #     if np.max(aimdata.loc[aim_date,'weekday'])==1:
    #         if np.max(aimdata.loc[i,'weekday']) ==1:
    #             onedayload = np.array(data_mm.loc[i, 'load'].values.flatten())
    #             onedayload = onedayload.reshape(48, 1)
    #             # print((onedayload * 0.45).shape, 5)
    #             allvalue.append(onedayload * 1.2)
    #         else:
    #             onedayload = np.array(data_mm.loc[i, 'load'].values.flatten())
    #             onedayload = onedayload.reshape(48, 1)
    #             # print((onedayload * 0.45).shape, 6)
    #             allvalue.append(onedayload * 0.6)
    #     if np.max(aimdata.loc[aim_date,'weekday'])==3:
    #         if np.max(aimdata.loc[i,'weekday']) ==3:
    #             onedayload = np.array(data_mm.loc[i, 'load'].values.flatten())
    #             onedayload = onedayload.reshape(48, 1)
    #             # print((onedayload * 0.45).shape, 7)
    #             allvalue.append(onedayload * 1.2)
    #         else:
    #             onedayload = np.array(data_mm.loc[i, 'load'].values.flatten())
    #             onedayload = onedayload.reshape(48, 1)
    #             # print((onedayload * 0.45).shape, 8)
    #             allvalue.append(onedayload * 0.6)

    # psfpre = sum(allvalue)/len(allvalue)
    # psfpre =sc_load.inverse_transform(psfpre.reshape(-1, 1)).flatten()
    return psfpre

def PSF_dayoutw3(csv_name, pred_date, data_index):
    cldata = pd.read_csv(csv_name, header=0, index_col=1, parse_dates=['id'])
    sday = []
    pred_date = datetime.datetime.strptime(pred_date, "%Y-%m-%d")
    sday.append(cldata.loc[pred_date, 'cluster_label'])
    sday.append(cldata.loc[pred_date + datetime.timedelta(days=-1), 'cluster_label'])
    sday.append(cldata.loc[pred_date + datetime.timedelta(days=-2), 'cluster_label'])
    print('window = 3')
    ps_index = np.array([])
    for i in data_index:
        if i + datetime.timedelta(days=3) in data_index:
            if cldata.loc[i, 'cluster_label'] == sday[2] and cldata.loc[
                i + datetime.timedelta(days=1), 'cluster_label'] == sday[1] and cldata.loc[
                i + datetime.timedelta(days=2), 'cluster_label'] == sday[0] and i + datetime.timedelta(
                days=3) != pred_date + datetime.timedelta(days=1):  # 分别代表 i当天模式与预测前三日 i+1天与预测前两日 i+2天与预测前一日 i+3天为待预测日
                ps_index = np.append(ps_index, i + datetime.timedelta(days=3))
    for i in data_index:
        if i + datetime.timedelta(days=-2) in data_index and i + datetime.timedelta(days=1) in data_index:
            if cldata.loc[i, 'cluster_label'] == sday[0] and cldata.loc[
                i + datetime.timedelta(days=-1), 'cluster_label'] == sday[1] and cldata.loc[
                i + datetime.timedelta(days=-2), 'cluster_label'] == sday[2] and i + datetime.timedelta(
                days=1) not in ps_index and i + datetime.timedelta(days=1) != pred_date + datetime.timedelta(
                days=1):
                ps_index = np.append(ps_index, i + datetime.timedelta(days=1))

    print('modelen=', len(ps_index))
    if len(ps_index) <= 2:
        print('window = 2')
        for i in data_index:
            if i + datetime.timedelta(days=2) in data_index:
                if cldata.loc[i, 'cluster_label'] == sday[1] and cldata.loc[
                    i + datetime.timedelta(days=1), 'cluster_label'] == sday[0] and i + datetime.timedelta(
                        days=2) != pred_date + datetime.timedelta(days=1):  # 分别代表 i当天模式与预测前2日 i+1天与预测前1日 i+2天与预测为待预测日
                    ps_index = np.append(ps_index, i + datetime.timedelta(days=2))
        for i in data_index:
            if i + datetime.timedelta(days=-1) in data_index and i + datetime.timedelta(days=1) in data_index:
                if cldata.loc[i, 'cluster_label'] == sday[0] and cldata.loc[
                    i + datetime.timedelta(days=-1), 'cluster_label'] == sday[1] and i + datetime.timedelta(
                        days=1) not in ps_index and i + datetime.timedelta(days=1) != pred_date + datetime.timedelta(
                        days=1):
                    ps_index = np.append(ps_index, i + datetime.timedelta(days=1))  # i当天与预测前1日 i-1天与预测前2日 i+1天待预测日

        print('modelen=', len(ps_index))
    return ps_index

def PSF_dayouttryw6(csv_name, pred_date, data_index):
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
    if len(ps_index)==0:
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
    if len(ps_index)==0:
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
    if len(ps_index) ==0:
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
    # if len(ps_index) == 0:
    #     print('window = 1')
    #     for i in data_index:
    #         if i + datetime.timedelta(days=1) in data_index:
    #             if cldata.loc[i, 'cluster_label'] == sday[0]  and i + datetime.timedelta(
    #                     days=1) != pred_date + datetime.timedelta(days=1):  # 分别代表 i当天模式与预测前2日 i+1天与预测前1日 i+2天与预测为待预测日
    #                 ps_index = np.append(ps_index, i + datetime.timedelta(days=1))
    #     for i in data_index:
    #         if i + datetime.timedelta(days=-1) in data_index:
    #             if cldata.loc[i + datetime.timedelta(days=-1), 'cluster_label'] == sday[0] and i  not in ps_index \
    #                     and i != pred_date + datetime.timedelta(days=1):
    #                 ps_index = np.append(ps_index, i)  # i当天与预测前1日 i-1天与预测前2日 i+1天待预测日

        # print('modelen=', len(ps_index))
    # data = pd.read_csv(csv_name, header=0, index_col=0, parse_dates=['date'])
    # weekday=pd.DataFrame(None)
    # for i in ps_index:
    #
    #         weekday = weekday.append(data.loc[i,'weekday'].any())


    return ps_index

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