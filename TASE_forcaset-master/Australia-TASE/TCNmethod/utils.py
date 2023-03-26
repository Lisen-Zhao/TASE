import torch
import numpy as np
from torch.autograd import Variable
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection

def seq_jiwen(dataname,psf_name):
    data = pd.read_csv(dataname, header=0, index_col=0, parse_dates=['date'])
    datapsf = pd.read_csv(psf_name, header=0, index_col=1, parse_dates=['date'])
    psf_index = datapsf.index.unique()
    pre_index = []
    for i in psf_index:
        pre_index.append(i + datetime.timedelta(days=-1))
    jiwenlist1 =[]
    jiwenlist2 =[]
    for i in pre_index:
        jiwen = 0
        for t in range(-30,1):
            jiwenday = (np.max(data.loc[i+ datetime.timedelta(days=t),'temp'])+np.min(data.loc[i+ datetime.timedelta(days=t),'temp']))/2
            jiwenday =  np.max([0,jiwenday-27])
            # jiwenday = jiwenday - 27
            jiwen= jiwenday+jiwen

        # ilist = np.stack(48 * (jiwen, jiwen), axis=0).reshape(-1)
        ilist = [jiwen]*48
        jiwenlist1.append(ilist)
    for i in pre_index:
        jiwen = 0
        for t in range(-29, 2):
            jiwenday = (np.max(data.loc[i + datetime.timedelta(days=t), 'temp']) + np.min(
                data.loc[i + +datetime.timedelta(days=t), 'temp'])) / 2
            jiwenday = np.max([0, jiwenday - 27])
            # jiwenday = jiwenday - 27
            jiwen = jiwenday + jiwen

        # ilist = np.stack(48 * (jiwen, jiwen), axis=0).reshape(-1)
        ilist = [jiwen] * 48
        jiwenlist2.append(ilist)

    return jiwenlist1,jiwenlist2
def data_generator(psf_name,dataname,jiwenlist_pre,jiwenlist_psf):
    """
    Args:
        seq_length: Length of the adding problem data
        N: # of data in the set
    """
    # X_num = torch.rand([N, 1, seq_length])
    # X_mask = torch.zeros([N, 1, seq_length])
    # Y = torch.zeros([N, 1])
    # for i in range(N):
    #     positions = np.random.choice(seq_length, size=2, replace=False)
    #     X_mask[i, 0, positions[0]] = 1
    #     X_mask[i, 0, positions[1]] = 1
    #     Y[i,0] = X_num[i, 0, positions[0]] + X_num[i, 0, positions[1]]
    # X = torch.cat((X_num, X_mask), dim=1)
    def reshape_data(datapsf, data, psf_index, pre_index,jiwenlist_pre,jiwenlist_psf):
        # jiwenlist_pre = jiwenlist_pre.reshape(-1)
        # jiwenlist_psf = jiwenlist_psf.reshape(-1)
        sc_hour = MinMaxScaler()
        data['hour'] = sc_hour.fit_transform(data['hour'].values.reshape(-1, 1))
        sc_load = MinMaxScaler()
        data['load'] = sc_load.fit_transform(data['load'].values.reshape(-1, 1))
        sc_temp = MinMaxScaler()
        data['temp'] = sc_temp.fit_transform(data['temp'].values.reshape(-1, 1))
        sc_jiwen = MinMaxScaler()
        jiwenlist_pre= sc_jiwen.fit_transform(np.array(jiwenlist_pre).reshape(-1, 1))
        jiwenlist_psf = sc_jiwen.fit_transform(np.array(jiwenlist_psf).reshape(-1, 1))
        # datapsf['PSF'] = sc_load.fit_transform(datapsf['PSF'].values.reshape(-1, 1))
        datapsf['PSF'] = datapsf['PSF'].values.reshape(-1, 1)
        # data = data.drop(data.columns[1], axis=1)  #删除温度部分 只剩下负荷和时间
        data_real = data.copy()
        data = data.drop(data.columns[2], axis=1)  # 删除负荷，只剩下时间

        length = psf_index.shape[0]
        # lengthtr = int(np.floor(length*0.8))
        x = np.column_stack((np.column_stack(( np.column_stack(( np.column_stack((
            data_real.loc[pre_index].iloc[:length * 48],jiwenlist_pre[:length * 48])), data.loc[psf_index].iloc[:length * 48])),
                                   datapsf.loc[psf_index, 'PSF'].iloc[:length * 48])),jiwenlist_psf[:length * 48])).reshape(-1, 96, 4)
        y = data_real.loc[psf_index, 'load'].iloc[:length * 48].values.reshape(-1, 48)
        # X_test = np.column_stack((np.column_stack((np.column_stack(( np.column_stack((
        #     data_real.loc[pre_index].iloc[length * 48:],jiwenlist_pre[length * 48:])), data.loc[psf_index].iloc[length * 48:])),
        #                           datapsf.loc[psf_index, 'PSF'].iloc[length * 48:])),jiwenlist_psf[length * 48:])).reshape(-1, 96, 4)
        # y_test = data_real.loc[psf_index, 'load'].iloc[length * 48:].values.reshape(-1, 48)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, shuffle=True, test_size=0.33)
        # X_train =  datapsf.loc[psf_index,'PSF'].iloc[:length*48].values.reshape(-1,48)
        # y_train = data.loc[psf_index,'load'].iloc[:length*48].values.reshape(-1,48)
        # X_test = datapsf.loc[psf_index,'PSF'].iloc[length*48:].values.reshape(-1,48)
        # y_test = data.loc[psf_index,'load'].iloc[length*48:].values.reshape(-1,48)
        print('        X_train.shape=', X_train.shape)
        return [X_train, X_test, y_train, y_test, sc_hour, sc_temp, sc_jiwen,sc_load]

    datapsf = pd.read_csv(psf_name, header=0, index_col=1, parse_dates=['date'])
    data = pd.read_csv(dataname, header=0, index_col=0, parse_dates=['date'])
    psf_index = datapsf.index.unique()
    data_index = data.index.unique()
    data = data.drop(data.columns[3], axis=1)  # 删除一列
    data = data.drop(data.columns[3], axis=1)  # 删除一列
    # data = data.drop(data.columns[3], axis=1)  # 删除一列
    pre_index = []
    for i in psf_index:
        pre_index.append(i + +datetime.timedelta(days=-1))
    a = reshape_data(datapsf, data, psf_index, pre_index,jiwenlist_pre,jiwenlist_psf)

    return torch.tensor(a[0]), torch.tensor(a[2]), torch.tensor(a[1]), torch.tensor(a[3]), a[-1], a[4], a[5],a[6]


def data_generatordeltaT(psf_name,dataname,temp1,temp2,temp3,temp4):
    """
    Args:
        seq_length: Length of the adding problem data
        N: # of data in the set
    """
    # X_num = torch.rand([N, 1, seq_length])
    # X_mask = torch.zeros([N, 1, seq_length])
    # Y = torch.zeros([N, 1])
    # for i in range(N):
    #     positions = np.random.choice(seq_length, size=2, replace=False)
    #     X_mask[i, 0, positions[0]] = 1
    #     X_mask[i, 0, positions[1]] = 1
    #     Y[i,0] = X_num[i, 0, positions[0]] + X_num[i, 0, positions[1]]
    # X = torch.cat((X_num, X_mask), dim=1)
    def reshape_data(datapsf, data, psf_index, pre_index,temp1,temp2,temp3,temp4):
        sc_hour = MinMaxScaler()
        data['hour'] = sc_hour.fit_transform(data['hour'].values.reshape(-1, 1))
        sc_load = MinMaxScaler()
        data['load'] = sc_load.fit_transform(data['load'].values.reshape(-1, 1))
        sc_temp = MinMaxScaler()
        data['temp'] = sc_temp.fit_transform(data['temp'].values.reshape(-1, 1))
        sc_tempdelta = MinMaxScaler()
        temp1 = sc_tempdelta.fit_transform(temp1.reshape(-1, 1))
        temp2 = sc_tempdelta.fit_transform(temp2.reshape(-1, 1))
        temp3 = sc_tempdelta.fit_transform(temp3.reshape(-1, 1))
        temp4 = sc_tempdelta.fit_transform(temp4.reshape(-1, 1))

        datapsf['PSF'] = sc_load.fit_transform(datapsf['PSF'].values.reshape(-1, 1))
        # data = data.drop(data.columns[1], axis=1)  #删除温度部分 只剩下负荷和时间
        data_real = data.copy()
        data = data.drop(data.columns[2], axis=1)  # 删除负荷，只剩下时间

        length = int(np.floor(psf_index.shape[0] * 0.8))
        # lengthtr = int(np.floor(length*0.8))
        X_train = np.column_stack((np.column_stack((np.column_stack((np.column_stack((np.column_stack((np.column_stack(
            (data_real.loc[pre_index].iloc[:length * 48],temp3[:length*48])),temp4[:length*48])), data.loc[psf_index].iloc[:length * 48])),
                                   datapsf.loc[psf_index, 'PSF'].iloc[:length * 48])),temp1[:length*48])),
                                                    temp2[:length*48])).reshape(-1, 96, 5)

        y_train = data_real.loc[psf_index, 'load'].iloc[:length * 48].values.reshape(-1, 48)
        X_test = np.column_stack((np.column_stack((np.column_stack((np.column_stack((np.column_stack((np.column_stack(
            (data_real.loc[pre_index].iloc[length * 48:],temp3[length*48:] )),temp4[length*48:])),data.loc[psf_index].iloc[length * 48:])),
                                  datapsf.loc[psf_index, 'PSF'].iloc[length * 48:])),temp1[length*48:])),
                                                    temp2[length*48:])).reshape(-1, 96, 5)
        y_test = data_real.loc[psf_index, 'load'].iloc[length * 48:].values.reshape(-1, 48)
        # X_train =  datapsf.loc[psf_index,'PSF'].iloc[:length*48].values.reshape(-1,48)
        # y_train = data.loc[psf_index,'load'].iloc[:length*48].values.reshape(-1,48)
        # X_test = datapsf.loc[psf_index,'PSF'].iloc[length*48:].values.reshape(-1,48)
        # y_test = data.loc[psf_index,'load'].iloc[length*48:].values.reshape(-1,48)
        print('        X_train.shape=', X_train.shape)
        return [X_train, X_test, y_train, y_test, sc_hour, sc_temp, sc_load]

    datapsf = pd.read_csv(psf_name, header=0, index_col=1, parse_dates=['date'])
    data = pd.read_csv(dataname, header=0, index_col=0, parse_dates=['date'])
    psf_index = datapsf.index.unique()
    data_index = data.index.unique()
    data = data.drop(data.columns[3], axis=1)  # 删除一列
    # data = data.drop(data.columns[3], axis=1)  # 删除一列
    pre_index = []
    for i in psf_index:
        pre_index.append(i + +datetime.timedelta(days=-1))
    a = reshape_data(datapsf, data, psf_index, pre_index,temp1,temp2,temp3,temp4)

    return torch.tensor(a[0]), torch.tensor(a[2]), torch.tensor(a[1]), torch.tensor(a[3]), a[-1], a[4], a[5]

def data_traindeltatemp(psf_name,dataname):
    data = pd.read_csv(dataname, header=0, index_col=0, parse_dates=['date'])
    datapsf = pd.read_csv(psf_name, header=0, index_col=1, parse_dates=['date'])
    psf_index = datapsf.index.unique()
    deltatemp1 = np.array([])
    deltatemp2 = np.array([])
    deltatemp3 = np.array([])
    deltatemp4 = np.array([])
    for i in psf_index:
        psftemp1= data.loc[i, 'temp'].values.flatten()
        psftemp2 = data.loc[i + datetime.timedelta(days=-1), 'temp'].values.flatten()
        psftemp3 = data.loc[i + datetime.timedelta(days=-2), 'temp'].values.flatten()
        psftemp4 = data.loc[i + datetime.timedelta(days=-3), 'temp'].values.flatten()

        deltatemp1 = np.append(deltatemp1, psftemp1 - psftemp2)
        deltatemp2 = np.append(deltatemp2, psftemp1 - psftemp3)
        deltatemp3 = np.append(deltatemp3, psftemp2 - psftemp3)
        deltatemp4 = np.append(deltatemp4, psftemp2 - psftemp4)
    return deltatemp1,deltatemp2,deltatemp3,deltatemp4


def data_deltatemp(data,aim_date):
    x_tpred = data.loc[aim_date, 'temp'].values.flatten()
    x_ytpred = data.loc[(aim_date + datetime.timedelta(days=-1)).strftime(
        "%Y-%m-%d"), 'temp'].values.flatten()
    x_yytpred = data.loc[(aim_date + datetime.timedelta(days=-2)).strftime(
        "%Y-%m-%d"), 'temp'].values.flatten()
    x_yytpred1 = data.loc[(aim_date + datetime.timedelta(days=-3)).strftime(
        "%Y-%m-%d"), 'temp'].values.flatten()
    predtemp1 = x_tpred - x_ytpred
    predtemp2 = x_tpred - x_yytpred
    predtemp3 = x_ytpred - x_yytpred
    predtemp4 = x_ytpred - x_yytpred1
    return predtemp1, predtemp2, predtemp3, predtemp4
