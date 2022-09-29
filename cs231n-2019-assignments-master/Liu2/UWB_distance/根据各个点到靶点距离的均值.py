import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlrd

lis = []
for i in range(1, 325):
    df = pd.read_excel('./dataset/abnormal/' + str(i) +'.异常.xls')
    X = df.iloc[:, 2:6]
    X_m = list(X.mean(axis=0))
    lis.append(X_m)

pd_list = pd.DataFrame(lis)
pd_list.to_csv('./dataset/abnormal_mean_distance.csv')
