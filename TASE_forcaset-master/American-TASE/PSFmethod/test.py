import pandas as pd
import datetime
import numpy as np
import PSF
import PSF_Inputdate

datacontent = '..\\CSVData\\datas.csv'
data = pd.read_csv(datacontent, header=0, index_col=0, parse_dates=['date'])

data = data.loc['2015-07-02':'2017-12-31']
data_index = data.index.unique()
# data = data.drop(data.columns[3], axis=1)  #删除一列
# data = data.drop(data.columns[3], axis=1)  #删除一列

PSF1 = []
psf_list=[]
nopsf=0
targ_list, aim_list = PSF_Inputdate.get_targ_aimlistw6(datacontent)

for i, j in zip(targ_list, aim_list):
    print(j)

    ps_index = PSF.PSF_dayouttryw6T('..\\CSVData\\cluster6LT.csv', i, data_index)
    # try:
    psfpre = PSF.PSF_result(data,datacontent,j, ps_index)
    PSF1.append(psfpre)
    psf_list.append(j)
    # except ZeroDivisionError:
    #     nopsf=nopsf+1
print('没有psf的天数有',nopsf,'天')

ilist = np.stack(12*(psf_list,psf_list), axis=1).reshape(-1)
PSF1 = np.array(PSF1).reshape(-1)
psf = pd.DataFrame({"date": ilist, "PSF": PSF1})

psf.to_csv('..\\CSVData\\psfdk=6w=6.csv')