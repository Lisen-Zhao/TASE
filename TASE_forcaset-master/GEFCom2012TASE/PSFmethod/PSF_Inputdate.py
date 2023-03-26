import pandas as pd
import datetime


def get_targ_aimlistw3(datacontent):
    data = pd.read_csv(datacontent, header=0, index_col=0, parse_dates=['date'])
    # data = data.loc['2004-7-1':'2007-10-30'].dropna()
    data = data.loc['2005-11-1': '2006-11-1'].dropna()
    data_index = data.index.unique()
    targ_list=[]
    aim_list=[]

    for i in data_index:
        # x = datetime.datetime.strptime(i, "%Y-%m-%d")
        if i+datetime.timedelta(days=-2) in data_index and i+datetime.timedelta(days=1) in data_index :
            targ_list.append(i)
            aim_list.append(i+datetime.timedelta(days=1))
    return targ_list, aim_list




def AUGtarg_aim_listw3(datacontent):
    data = pd.read_csv(datacontent, header=0, index_col=0, parse_dates=['date'])
    data = data.loc['2008-05-31':'2008-06-28'].dropna()
    data_index = data.index.unique()
    targ_list=[]
    aim_list=[]

    for i in data_index:
        # x = datetime.datetime.strptime(i, "%Y-%m-%d")

            targ_list.append(i)
            aim_list.append(i+datetime.timedelta(days=1))
    return targ_list, aim_list

def AUGtarg_aim_list0711(datacontent):
    data = pd.read_csv(datacontent, header=0, index_col=0, parse_dates=['date'])
    data = data.loc['2007-10-31':'2007-11-29'].dropna()
    data_index = data.index.unique()
    targ_list=[]
    aim_list=[]

    for i in data_index:
        # x = datetime.datetime.strptime(i, "%Y-%m-%d")

            targ_list.append(i)
            aim_list.append(i+datetime.timedelta(days=1))
    return targ_list, aim_list

def AUGtarg_aim_listw4(datacontent):
    data = pd.read_csv(datacontent, header=0, index_col=0, parse_dates=['date'])
    data = data.loc['2008-05-31':'2008-06-28'].dropna()
    data_index = data.index.unique()
    targ_list=[]
    aim_list=[]

    for i in data_index:
        # x = datetime.datetime.strptime(i, "%Y-%m-%d")

            targ_list.append(i)
            aim_list.append(i+datetime.timedelta(days=1))
    return targ_list, aim_list

def FEBtarg_aim_list0802(datacontent):
    data = pd.read_csv(datacontent, header=0, index_col=0, parse_dates=['date'])
    data = data.loc['2008-01-31':'2008-02-27'].dropna()
    data_index = data.index.unique()
    targ_list=[]
    aim_list=[]

    for i in data_index:
        # x = datetime.datetime.strptime(i, "%Y-%m-%d")

            targ_list.append(i)
            aim_list.append(i+datetime.timedelta(days=1))
    return targ_list, aim_list

def MAYtarg_aim_list0805(datacontent):
    data = pd.read_csv(datacontent, header=0, index_col=0, parse_dates=['date'])
    data = data.loc['2008-04-30':'2008-05-30'].dropna()
    data_index = data.index.unique()
    targ_list=[]
    aim_list=[]

    for i in data_index:
        # x = datetime.datetime.strptime(i, "%Y-%m-%d")

            targ_list.append(i)
            aim_list.append(i+datetime.timedelta(days=1))
    return targ_list, aim_list

def targ_aim_list0806(datacontent):
    data = pd.read_csv(datacontent, header=0, index_col=0, parse_dates=['date'])
    data = data.loc['2008-05-31':'2008-06-28'].dropna()
    data_index = data.index.unique()
    targ_list=[]
    aim_list=[]

    for i in data_index:
        # x = datetime.datetime.strptime(i, "%Y-%m-%d")

            targ_list.append(i)
            aim_list.append(i+datetime.timedelta(days=1))
    return targ_list, aim_list

def AUGtarg_aim_listw5(datacontent):
    data = pd.read_csv(datacontent, header=0, index_col=0, parse_dates=['date'])
    data = data.loc['2004-07-01':'2004-07-30'].dropna()
    data_index = data.index.unique()
    targ_list=[]
    aim_list=[]

    for i in data_index:
        # x = datetime.datetime.strptime(i, "%Y-%m-%d")

            targ_list.append(i)
            aim_list.append(i+datetime.timedelta(days=1))
    return targ_list, aim_list