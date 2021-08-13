import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor, StackingRegressor
from lightgbm import LGBMRegressor

def fitnessfunction(vardim, param, bound, data):
    # data import
    StandData = data

    # Divide data set
    X = StandData.iloc[:, 0:30]
    Y = StandData.iloc[:, -1]

    from sklearn.model_selection import RepeatedKFold
    KF = RepeatedKFold(n_splits=10, n_repeats=1, random_state=7)
    MSE = []
    models = [('RF', RandomForestRegressor(n_estimators=param[0].astype(int), max_depth=param[1].astype(int))),('BagDT', BaggingRegressor(n_estimators=param[2].astype(int))),('LGB', LGBMRegressor(learning_rate=param[3], n_estimators=param[4].astype(int)))]

    clf = StackingRegressor(estimators=models, final_estimator=LinearRegression())

    for train_index, test_index in KF.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        clf.fit(X_train, Y_train)
        pred_Y = clf.predict(X_test)
        MSE.append(mean_squared_error(Y_test, pred_Y))
    AveMSE = np.mean(MSE)
    print('MSE is:', AveMSE)
    return AveMSE*(-1.0)
