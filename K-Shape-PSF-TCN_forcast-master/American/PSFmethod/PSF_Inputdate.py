import pandas as pd
import datetime


def get_targ_aimlistw6(datacontent):
    data = pd.read_csv(datacontent, header=0, index_col=0, parse_dates=['date'])
    data = data.loc['2018-01-01':'2018-12-31']
    data_index = data.index.unique()
    targ_list=[]
    aim_list=[]

    for i in data_index:
        # x = datetime.datetime.strptime(i, "%Y-%m-%d")
        if i+datetime.timedelta(days=-5) in data_index and i+datetime.timedelta(days=1) in data_index :
            targ_list.append(i)
            aim_list.append(i+datetime.timedelta(days=1))
    return targ_list, aim_list

def novtarg_aim_listw6(datacontent):
    data = pd.read_csv(datacontent, header=0, index_col=0, parse_dates=['date'])
    data = data.loc['2021-04-30':'2021-05-30'].dropna()
    data_index = data.index.unique()
    targ_list=[]
    aim_list=[]

    for i in data_index:
        # x = datetime.datetime.strptime(i, "%Y-%m-%d")

            targ_list.append(i)
            aim_list.append(i+datetime.timedelta(days=1))
    return targ_list, aim_list

def novtarg_aim_listw7(datacontent):
    data = pd.read_csv(datacontent, header=0, index_col=0, parse_dates=['date'])
    data = data.loc['2021-07-31':'2021-08-30'].dropna()
    data_index = data.index.unique()
    targ_list=[]
    aim_list=[]

    for i in data_index:
        # x = datetime.datetime.strptime(i, "%Y-%m-%d")

            targ_list.append(i)
            aim_list.append(i+datetime.timedelta(days=1))
    return targ_list, aim_list

def novtarg_aim_listw8(datacontent):
    data = pd.read_csv(datacontent, header=0, index_col=0, parse_dates=['date'])
    data = data.loc['2022-01-31':'2022-02-27'].dropna()
    data_index = data.index.unique()
    targ_list=[]
    aim_list=[]

    for i in data_index:
        # x = datetime.datetime.strptime(i, "%Y-%m-%d")

            targ_list.append(i)
            aim_list.append(i+datetime.timedelta(days=1))
    return targ_list, aim_list

def novtarg_aim_listw9(datacontent):
    data = pd.read_csv(datacontent, header=0, index_col=0, parse_dates=['date'])
    data = data.loc['2021-10-31':'2021-11-29'].dropna()
    data_index = data.index.unique()
    targ_list=[]
    aim_list=[]

    for i in data_index:
        # x = datetime.datetime.strptime(i, "%Y-%m-%d")

            targ_list.append(i)
            aim_list.append(i+datetime.timedelta(days=1))
    return targ_list, aim_list



def maytarg_aim_listw6(datacontent):
    data = pd.read_csv(datacontent, header=0, index_col=0, parse_dates=['date'])
    data = data.loc['2010-04-30':'2010-05-30'].dropna()
    data_index = data.index.unique()
    targ_list=[]
    aim_list=[]

    for i in data_index:
        # x = datetime.datetime.strptime(i, "%Y-%m-%d")

            targ_list.append(i)
            aim_list.append(i+datetime.timedelta(days=1))
    return targ_list, aim_list