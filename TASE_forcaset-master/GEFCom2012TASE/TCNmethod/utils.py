import torch
import numpy as np
from torch.autograd import Variable
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
def seq_jiwen(dataname,psf_name,tempdata):
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
        for t in range(-7,1):
            jiwenday = (np.max(tempdata.loc[i+ datetime.timedelta(days=t)])+np.min(tempdata.loc[i+ datetime.timedelta(days=t)]))/2
            jiwenday =  np.max([0,jiwenday-88])
            # jiwenday = jiwenday - 27
            jiwen= jiwenday+jiwen

        # ilist = np.stack(48 * (jiwen, jiwen), axis=0).reshape(-1)
        ilist = [jiwen]*24
        jiwenlist1.append(ilist)
    for i in pre_index:
        jiwen = 0
        for t in range(-6, 2):
            jiwenday = (np.max(tempdata.loc[i + datetime.timedelta(days=t)]) + np.min(
                tempdata.loc[i + +datetime.timedelta(days=t)])) / 2
            jiwenday = np.max([0, jiwenday - 88])
            # jiwenday = jiwenday - 27
            jiwen = jiwenday + jiwen

        # ilist = np.stack(48 * (jiwen, jiwen), axis=0).reshape(-1)
        ilist = [jiwen] * 24
        jiwenlist2.append(ilist)

    return jiwenlist1,jiwenlist2

def data_generator(psf_name,data,tempdata,jiwenlist_pre,jiwenlist_psf ):
    """
    Args:
        seq_length: Length of the adding problem data
        N: # of data in the set
    """
    def reshape_data(datapsf, data, tempdata, psf_index, pre_index,jiwenlist_pre,jiwenlist_psf ):
        # sc_hour = MinMaxScaler()
        # data['hour'] = sc_hour.fit_transform(data['hour'].values.reshape(-1, 1))
        # sc_load = StandardScaler()
        # sc_load2 = StandardScaler()
        # sc_temp = StandardScaler()
        sc_load = MinMaxScaler()
        sc_load2 = MinMaxScaler()
        sc_temp = MinMaxScaler()
        load_train_pre = pd.DataFrame(sc_load.fit_transform(data.loc[pre_index].values.reshape(-1, 1)).reshape(-1, 24))
        load_test = pd.DataFrame(sc_load.transform(data.loc[psf_index].values.reshape(-1, 1)).reshape(-1, 24))
        temp_train_pre = pd.DataFrame(sc_temp.fit_transform(tempdata.loc[pre_index].values.reshape(-1, 1)).reshape(-1, 24))
        temp_train = pd.DataFrame(sc_temp.transform(tempdata.loc[psf_index].values.reshape(-1, 1)).reshape(-1, 24))
        sc_jiwen = MinMaxScaler()
        jiwenlist_pre = sc_jiwen.fit_transform(np.array(jiwenlist_pre).reshape(-1, 1))
        jiwenlist_psf = sc_jiwen.fit_transform(np.array(jiwenlist_psf).reshape(-1, 1))
        load_train_pre.index, load_train_pre.columns = pre_index, data.columns
        temp_train_pre.index, temp_train_pre.columns = pre_index, data.columns
        temp_train.index = psf_index
        dataxpsf = datapsf.copy()
        dataxpsf['PSF'] = sc_load.transform(dataxpsf['PSF'].values.reshape(-1, 1))
        dataxpsf['PSF'] = dataxpsf['PSF'].values.reshape(-1, 1)
        length = int(np.floor(psf_index.shape[0]))
        x=np.column_stack((np.column_stack((np.column_stack((np.column_stack((np.column_stack((
            load_train_pre.iloc[:length].values.reshape(-1,1), temp_train_pre.iloc[:length].values.reshape(-1,1))),jiwenlist_pre[:length * 24])),

                                            dataxpsf.loc[psf_index, 'PSF'].iloc[:length * 24].values.reshape(-1, 1))),
                            temp_train.iloc[:length].values.reshape(-1, 1))),jiwenlist_pre[:length * 24])).reshape(-1, 48, 3)
        y = load_test.iloc[:length].values.reshape(-1, 24)
        X_train,X_test,y_train,y_test = model_selection.train_test_split(x,y,shuffle=True,test_size=0.33)
        # X_train = np.column_stack((load_train_pre.iloc[:length].values.reshape(-1,1), temp_train_pre.iloc[:length].values.reshape(-1,1)))
        # X_train = np.column_stack((X_train,dataxpsf.loc[psf_index, 'PSF'].iloc[:length * 24].values.reshape(-1,1)))
        # X_train = np.column_stack((X_train,temp_train.iloc[:length].values.reshape(-1,1))).reshape(-1, 48, 2)
        # y_train = load_test.iloc[:length].values.reshape(-1, 24)
        #
        # X_test = np.column_stack((load_train_pre.iloc[length:].values.reshape(-1, 1), temp_train_pre.iloc[length:].values.reshape(-1, 1)))
        # X_test = np.column_stack((X_test, dataxpsf.loc[psf_index, 'PSF'].iloc[length * 24:].values.reshape(-1,1)))
        # X_test = np.column_stack((X_test, temp_train.iloc[length:].values.reshape(-1, 1))).reshape(-1, 48, 2)
        # y_test = load_test.iloc[length:].values.reshape(-1, 24)

        print('        X_train.shape=', X_train.shape)
        return [X_train, X_test, y_train, y_test, sc_temp,sc_jiwen,  sc_load]

    datapsf = pd.read_csv(psf_name, header=0, index_col=1, parse_dates=['date'])
    # data = pd.read_csv(data_name, header=0, index_col=0, parse_dates=['date'])
    # tempdata = pd.read_csv(tempname, header=0, index_col=0, parse_dates=['date'])
    psf_index = datapsf.index.unique()
    # data_index = data.index.unique()
    # data = data.drop(data.columns[3], axis=1)  # 删除一列
    # data = data.drop(data.columns[3], axis=1)  # 删除一列
    pre_index = []
    for i in psf_index:
        pre_index.append(i + +datetime.timedelta(days=-1))
    a = reshape_data(datapsf, data, tempdata, psf_index, pre_index,jiwenlist_pre,jiwenlist_psf )

    return torch.tensor(a[0]), torch.tensor(a[2]), torch.tensor(a[1]), torch.tensor(a[3]), a[-1], a[4],a[5]


def data_generatordeltaT(psf_name,data,tempdata,temp1,temp2,temp3,temp4):
    """
    Args:
        seq_length: Length of the adding problem data
        N: # of data in the set
    """

    def reshape_data(datapsf, data,tempdata,psf_index, pre_index,temp1,temp2,temp3,temp4):
        # psf_index = datapsf.index.unique()
        sc_load = StandardScaler()
        sc_temp = StandardScaler()
        sc_delta = StandardScaler()
        load_train_pre = pd.DataFrame(sc_load.fit_transform(data.loc[pre_index].values.reshape(-1, 1)).reshape(-1, 24))
        load_test = pd.DataFrame(sc_load.fit_transform(data.loc[psf_index].values.reshape(-1, 1)).reshape(-1, 24))
        temp_train_pre = pd.DataFrame(
            sc_temp.fit_transform(tempdata.loc[pre_index].values.reshape(-1, 1)).reshape(-1, 24))
        temp_train = pd.DataFrame(sc_temp.fit_transform(tempdata.loc[psf_index].values.reshape(-1, 1)).reshape(-1, 24))
        load_train_pre.index, load_train_pre.columns = pre_index, data.columns
        temp_train_pre.index, temp_train_pre.columns = pre_index, data.columns
        temp_train.index = psf_index
        dataxpsf = datapsf.copy()
        dataxpsf['PSF'] = sc_load.fit_transform(dataxpsf['PSF'].values.reshape(-1, 1))
        temp1 = sc_delta.fit_transform(temp1.reshape(-1, 1))
        temp2 = sc_delta.fit_transform(temp2.reshape(-1, 1))
        temp3 = sc_delta.fit_transform(temp3.reshape(-1, 1))
        temp4 = sc_delta.fit_transform(temp4.reshape(-1, 1))

        length = int(np.floor(psf_index.shape[0] * 0.8))

        X_train = np.column_stack(
            (load_train_pre.iloc[:length].values.reshape(-1, 1), temp_train_pre.iloc[:length].values.reshape(-1, 1)))
        X_train = np.column_stack((np.column_stack((X_train, temp3[:length*24])),temp4[:length*24]))
        X_train = np.column_stack((X_train, dataxpsf.loc[psf_index, 'PSF'].iloc[:length * 24].values.reshape(-1, 1)))
        X_train = np.column_stack((X_train, temp_train.iloc[:length].values.reshape(-1, 1)))
        X_train = np.column_stack((np.column_stack((X_train, temp1[:length * 24])), temp2[:length * 24])).reshape(-1, 48, 4)
        y_train = load_test.iloc[:length].values.reshape(-1, 24)

        X_test = np.column_stack(
            (load_train_pre.iloc[length:].values.reshape(-1, 1), temp_train_pre.iloc[length:].values.reshape(-1, 1)))
        X_test = np.column_stack((np.column_stack((X_test, temp3[length * 24:])), temp4[length * 24:]))
        X_test = np.column_stack((X_test, dataxpsf.loc[psf_index, 'PSF'].iloc[length * 24:].values.reshape(-1, 1)))
        X_test = np.column_stack((X_test, temp_train.iloc[length:].values.reshape(-1, 1)))
        X_test = np.column_stack((np.column_stack((X_test, temp1[length * 24:])), temp2[length * 24:])).reshape(-1, 48, 4)
        y_test = load_test.iloc[length:].values.reshape(-1, 24)

        print('X_train.shape=', X_train.shape)
        return [X_train, X_test, y_train, y_test, sc_temp,sc_delta, sc_load]

    datapsf = pd.read_csv(psf_name, header=0, index_col=1, parse_dates=['date'])
    psf_index = datapsf.index.unique()
    data_index = data.index.unique()
    # data = data.drop(data.columns[3], axis=1)  # 删除一列
    # data = data.drop(data.columns[3], axis=1)  # 删除一列
    pre_index = []
    for i in psf_index:
        pre_index.append(i + +datetime.timedelta(days=-1))
    a = reshape_data(datapsf, data,tempdata, psf_index, pre_index,temp1,temp2,temp3,temp4)

    return torch.tensor(a[0]), torch.tensor(a[2]), torch.tensor(a[1]), torch.tensor(a[3]), a[-1], a[4], a[5]
def data_traindeltatemp(psf_name,dataname,tempname):
    data = pd.read_csv(dataname, header=0, index_col=0, parse_dates=['date'])
    # data = data.drop(data.columns[0], axis=1)  # 删除一列
    datapsf = pd.read_csv(psf_name, header=0, index_col=1, parse_dates=['date'])
    tempdata = pd.read_csv(tempname, header=0, index_col=0, parse_dates=['date'])
    psf_index = datapsf.index.unique()
    deltatemp1 = np.array([])
    deltatemp2 = np.array([])
    deltatemp3 = np.array([])
    deltatemp4 = np.array([])
    for i in psf_index:
        psftemp1= tempdata.loc[i].values.flatten()
        psftemp2 = tempdata.loc[i + datetime.timedelta(days=-1)].values.flatten()
        psftemp3 = tempdata.loc[i + datetime.timedelta(days=-2)].values.flatten()
        psftemp4 = tempdata.loc[i + datetime.timedelta(days=-3)].values.flatten()

        deltatemp1 = np.append(deltatemp1, psftemp1 - psftemp2)
        deltatemp2 = np.append(deltatemp2, psftemp1 - psftemp3)
        deltatemp3 = np.append(deltatemp3, psftemp2 - psftemp3)
        deltatemp4 = np.append(deltatemp4, psftemp2 - psftemp4)
    return deltatemp1,deltatemp2,deltatemp3,deltatemp4


def data_deltatemp(data, aim_date, sc_delta):
    # data = data.drop(data.columns[0], axis=1)  # 删除一列
    x_tpred = data.loc[aim_date].values.flatten()
    x_ytpred = data.loc[aim_date + datetime.timedelta(days=-1)].values.flatten()
    x_yytpred = data.loc[aim_date + datetime.timedelta(days=-2)].values.flatten()
    x_yytpred1 = data.loc[aim_date + datetime.timedelta(days=-3)].values.flatten()
    predtemp1 = (x_tpred - x_ytpred)
    predtemp2 = (x_tpred - x_yytpred)
    predtemp3 = (x_ytpred - x_yytpred)
    predtemp4 = (x_ytpred - x_yytpred1)
    predtemp1 = sc_delta.fit_transform(predtemp1.reshape(-1, 1))
    predtemp2 = sc_delta.fit_transform(predtemp2.reshape(-1, 1))
    predtemp3 = sc_delta.fit_transform(predtemp3.reshape(-1, 1))
    predtemp4 = sc_delta.fit_transform(predtemp4.reshape(-1, 1))
    return predtemp1, predtemp2, predtemp3, predtemp4
