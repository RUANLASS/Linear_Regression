import pandas as pd
import numpy as np
import math
def create_data():
    df = pd.read_csv("housing.csv")
    df2 = df.iloc[:, :-1]
    cols_list = list(df2.columns)
    cols_list.pop(-1)
    for i in cols_list:
        df2[f'{i}_squared']=df2[i]**2
        df2[f'{i}_cubed']=df2[i]**3
    
    y_col = df2.pop("median_house_value")
    df2["median_house_value"] = y_col
    rows = df2.shape[0]
    x = df2.iloc[:,:-1]
    y = df2.iloc[:,-1]
    train_boundary=math.floor(0.6*rows)
    cv_boundary = math.floor(0.8*rows)
    x_train = df2.iloc[0:train_boundary, :-1]
    y_train = df2.iloc[0:train_boundary,-1]
    x_cv = df2.iloc[train_boundary:cv_boundary,:-1]
    y_cv = df2.iloc[train_boundary:cv_boundary,-1]
    x_test = df2.iloc[cv_boundary:rows,:-1]
    y_test = df2.iloc[cv_boundary:rows,-1]
    
    return x,y,x_train,y_train,x_cv,y_cv,x_test,y_test


'''
1. Implement regularization
2. split into training, test, cv sets
3. Implement J(train), J(test), J(cv) diagnostics
4. Run diagnostics, fix predictions. 
'''