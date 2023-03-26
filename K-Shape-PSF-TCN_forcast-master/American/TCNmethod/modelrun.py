import torch
import argparse
import torch.optim as optim
import torch.nn.functional as F
import sys
sys.path.append("../../")
from model import TCN
from utils import data_generator,data_traindeltatemp,data_generatordeltaT,data_deltatemp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
import PSFmethod.PSF_Inputdate as PSF_Inputdate
import PSFmethod.PSF as PSF

model = torch.load('3.28_2.tar')
# model = torch.load('best_model.tar')
psftrainfilename = '..\\CSVData\\psfdk=6w=6.csv'
psffilename = '..\\CSVData\\cluster6LT.csv'
dataname = '..\\CSVData\\datas.csv'
savefigname = 'kshapetcnwin.jpg'
savecsvname = 'wintermape.csv'

X_train, Y_train, X_test, Y_test, sc_load, sc_temp = data_generator(psftrainfilename,dataname)

targ_list, aim_list = PSF_Inputdate.novtarg_aim_listw6(dataname)
# targ_list = ['2010-04-21']
# aim_list = ['2010-04-22']
# targ_list = ['2010-09-08','2010-09-23','2010-10-24','2010-11-01','2010-11-06']
# aim_list = ['2010-09-09','2010-09-24','2010-10-25','2010-11-02','2010-11-07']
data_real = pd.read_csv(dataname, header=0, index_col=0, parse_dates=['date'])

# data_real = data_real.drop(data_real.columns[3], axis=1)
data_real = data_real.drop(data_real.columns[4], axis=1)
data_real = data_real.drop(data_real.columns[4], axis=1)
data_real1 = data_real.copy()
data = data_real.copy()

data_real1 = data_real1.drop(data_real1.columns[0], axis=1)
# data_real['hour'] = sc_hour.fit_transform(data_real['hour'].values.reshape(-1, 1))
data_real['load'] = sc_load.transform(data_real['load'].values.reshape(-1, 1))
data_real['temp1'] = sc_temp.transform(data_real['temp1'].values.reshape(-1, 1))
data_real['temp2'] = sc_temp.transform(data_real['temp2'].values.reshape(-1, 1))
data_real['temp3'] = sc_temp.transform(data_real['temp3'].values.reshape(-1, 1))
data_real1['temp1'] = sc_temp.transform(data_real1['temp1'].values.reshape(-1, 1))
data_real1['temp2'] = sc_temp.transform(data_real1['temp2'].values.reshape(-1, 1))
data_real1['temp3'] = sc_temp.transform(data_real1['temp3'].values.reshape(-1, 1))
# data_real1['hour'] = sc_hour.fit_transform(data_real1['hour'].values.reshape(-1, 1))
# data_real = data_real.drop(data_real.columns[1], axis=1)
# data = data.drop(data.columns[1], axis=1)
data_index = data_real.loc['2019-01-01':'2020-12-31'].index.unique()
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
        psfpre = sc_load.transform(psfpre.reshape(-1,1))
    # predtemp1,predtemp2,predtemp3,predtemp4 = data_deltatemp(data_real, j)


        for_pred = np.column_stack((np.column_stack((data_real.loc[i],data_real1.loc[j])),psfpre)).reshape(-1,48,4)      #把前一天数据变成二维形式跑模型得到预测值preddata

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
# plt.show()
plt.savefig(savefigname)
print(mape_values)
pd.DataFrame(mape_values, index=ok_list).to_csv(savecsvname)
pd.DataFrame(pred.reshape(-1,24), columns=[str(x) + 'h' for x in range(1,25)], index=aim_list).to_csv('win_prevalue.csv')