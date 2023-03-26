import torch
import argparse
import torch.optim as optim
import torch.nn.functional as F
import sys
sys.path.append("../../")
from model import TCN
from utils import data_generator,seq_jiwen,data_traindeltatemp,data_generatordeltaT,data_deltatemp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
import PSFmethod.PSF_Inputdate as PSF_Inputdate
import PSFmethod.PSF as PSF

model = torch.load('Paper Model.tar')

psftrainfilename = '..\\CSVData\\psfdk=6w=6.csv'
psffilename = '..\\CSVData\\cluster6LT.csv'
dataname = '..\\CSVData\\datas.csv'
jiwenlist_pre,jiwenlist_psf = seq_jiwen(dataname,psftrainfilename)
X_train, Y_train, X_test, Y_test, sc_load, sc_hour, sc_temp,sc_jiwen = data_generator(psftrainfilename,dataname,jiwenlist_pre,jiwenlist_psf)

targ_list, aim_list = PSF_Inputdate.novtarg_aim_listw7(dataname)
# targ_list = ['2010-04-21']
# aim_list = ['2010-04-22']
# targ_list = ['2010-09-08','2010-09-23','2010-10-24','2010-11-01','2010-11-06']
# aim_list = ['2010-09-09','2010-09-24','2010-10-25','2010-11-02','2010-11-07']
data_real = pd.read_csv(dataname, header=0, index_col=0, parse_dates=['date'])

# data_real = data_real.drop(data_real.columns[3], axis=1)
data_real = data_real.drop(data_real.columns[3], axis=1)
data_real = data_real.drop(data_real.columns[3], axis=1)
data_real1 = data_real.copy()
data = data_real.copy()
data_real1 = data_real1.drop(data_real1.columns[2], axis=1)
data_real['hour'] = sc_hour.fit_transform(data_real['hour'].values.reshape(-1, 1))
data_real['load'] = sc_load.fit_transform(data_real['load'].values.reshape(-1, 1))
data_real['temp'] = sc_temp.fit_transform(data_real['temp'].values.reshape(-1, 1))
data_real1['temp'] = sc_temp.fit_transform(data_real1['temp'].values.reshape(-1, 1))
data_real1['hour'] = sc_hour.fit_transform(data_real1['hour'].values.reshape(-1, 1))
# data_real = data_real.drop(data_real.columns[1], axis=1)
# data = data.drop(data.columns[1], axis=1)
data_index = data_real.loc['2009-01-01':'2009-12-31'].index.unique()
pred = np.array([])
true = np.array([])
mape_values = []
PSF1 = []
ok_list=[]
nopsf=0
for i, j in zip(targ_list, aim_list):
    # i = datetime.datetime.strptime(i, "%Y-%m-%d")
    try:
        ps_index = PSF.PSF_dayouttryw6T(psffilename, i, data_index)
        psfpre = PSF.PSF_result(data_real, dataname,j,ps_index)
        ok_list.append(j)

    # print('没有psf的天数有', nopsf, '天')
    # psfpre = sc_load.fit_transform(psfpre.reshape(-1,1))
    # predtemp1,predtemp2,predtemp3,predtemp4 = data_deltatemp(data_real, j)

        jiwen = 0
        for t in range(-30, 1):  # -7,1
            jiwenday = (np.max(data.loc[i + datetime.timedelta(days=t), 'temp']) + np.min(
                data.loc[i + datetime.timedelta(days=t), 'temp'])) / 2
            jiwenday = np.max([0, jiwenday - 30])
            jiwen = jiwenday + jiwen
        ilist1 = [jiwen] * 48
        jiwen = 0
        for t in range(-29, 2):  # -6,2
            jiwenday = (np.max(data.loc[i + datetime.timedelta(days=t), 'temp']) + np.min(
                data.loc[i + datetime.timedelta(days=t), 'temp'])) / 2
            jiwenday = np.max([0, jiwenday - 30])
            jiwen = jiwenday + jiwen
        ilist2 = [jiwen] * 48
        ilist1 = sc_jiwen.fit_transform(np.array(ilist1).reshape(-1, 1))
        ilist2 = sc_jiwen.fit_transform(np.array(ilist2).reshape(-1, 1))
        for_pred = np.column_stack((np.column_stack(
            (np.column_stack((np.column_stack((data_real.loc[i], ilist1)), data_real1.loc[j])), psfpre)),
                                    ilist2)).reshape(-1, 96, 4)  # 把前一天数据变成二维形式跑模型得到预测值preddata
        # print(for_pred)
        pred_data = model(torch.tensor(for_pred).float().cuda())
        pred_data = np.array((list(pred_data.cpu().flatten().detach())))
        pred_data_true = sc_load.inverse_transform(pred_data.reshape(-1, 1)).flatten()   #逆归一化 得到风速数据 逆归一化只支持2维

        true_value = data.loc[j,'load'].values.flatten()
        print(i, 'to predict', j)
        # plt.plot(psfpre,'g')
        plt.plot(pred_data_true, 'b')
        plt.plot(true_value,'r')
        # plt.show()
        mapes = np.mean(np.abs(true_value - pred_data_true) / true_value) * 100
        print('mape:', mapes)
        pred = np.append(pred, pred_data_true)
        true = np.append(true, true_value)
        mape_values.append(mapes)
    except ZeroDivisionError:
        nopsf = nopsf + 1
print(np.mean(mape_values))
plt.figure(figsize=(30,5))
plt.plot(pred,'b')
plt.plot(true,'r')
plt.savefig('kshapetcnsum777.jpg')
print(mape_values)
pd.DataFrame(mape_values, index=ok_list).to_csv('mape对比老突变日sum777.csv')
pd.DataFrame(pred.reshape(-1,48), columns=[str(x) + 'h' for x in range(1,49)], index=aim_list).to_csv('predjiwensum.csv')