import pandas as pd
import openpyxl
import numpy as np
import csv
import datetime as dt

from matplotlib import pyplot as plt
from pandas.core.reshape import encoding
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

# label encoding for categorical vars
title = pd.read_csv("C:\\Users\\Mahmo\\OneDrive\\Desktop\\train.csv", encoding='ISO-8859-1')
encoding = LabelEncoder()
title['Title - F1'] = encoding.fit_transform(title['Title - F1'])
title['Category - F2'] = encoding.fit_transform(title['Category - F2'])
# data cleaning for the missed values
imputes = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(title)
x = title.iloc[:, :1]
y = title['Category - F2']
print(x.head())
print(y.head())

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=33)
# 1st Linear regression model
# model = LinearRegression()
# 2nd sklearn module for lasso regression
# model = Lasso(alpha=1.0)
# 3rd sklearn module for  ridge regression
# model = Ridge(alpha=0.1)
# 4th sklearn module for elastic net regression
# model = ElasticNet(alpha=0.05)
# 5th sklearn module for random forest model
model = RandomForestRegressor()
# 6th sklearn module for gredient boosting regressor model
# model = GradientBoostingRegressor(learning_rate=0.05)
model.fit(x_train, y_train)
y_predict = model.predict(x_train)
print('predict values:', y_predict)
mse = mean_squared_error(y_train, y_predict, multioutput='uniform_average')
print('mean square error:', np.sqrt(mse))

################################################TEST###############################################
title2 = pd.read_csv("C:\\Users\Mahmo\\OneDrive\\Desktop\\test.csv", encoding='ISO-8859-1')
# print(title2.head())
x1 = title2["Title - F1"]
title2['Title - F1'] = encoding.fit_transform(title2['Title - F1'])
imputes2 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(title2)
y1 = title2.iloc[:, :1]
y_predict2 = model.predict(y1)
final = pd.DataFrame({'titles': x1, 'Category': y_predict2})
print(final)
final.to_csv('C:\\Users\\Mahmo\\OneDrive\\Desktop\\predictions.csv')
print('sucess')


################################################graph###############################################
def plotGraph(y_train, y_pred_train, rand):
    if max(y_train) >= max(y_pred_train):
        my_range = int(max(y_train))
    else:
        my_range = int(max(y_pred_train))
    plt.scatter(range(len(y_train)), y_train, color='blue')
    plt.scatter(range(len(y_pred_train)), y_pred_train, color='red')
    plt.title(rand)
    plt.show()
    return


plotGraph(y_train, y_predict, 'random forest')
