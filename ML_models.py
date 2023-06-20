import utils
from utils import *

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score

from scipy.stats import pearsonr

def rfr(X, y, test_size=0.1, random_state = 0, iterations = 5):
    r2_mean_list = []
    MSE_mean_list = []
    pearson_mean_list = []
    for i in range(iterations):
        print(f'Iteration: {i+1}')
        X_train, X_val, y_train, y_val = cell_line_split(X, y, test_size, random_state) #train-validation split
        classify = RandomForestRegressor(n_jobs=-1, max_depth=300, n_estimators=200)
        classify.fit(X_train.values, y_train)
        y_pred = classify.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        MSE = mean_squared_error(y_val.values, y_pred)
        pearson = pearsonr(y_val.values, y_pred)
        r2_mean_list.append(r2)
        MSE_mean_list.append(MSE)
        pearson_mean_list.append(pearson[0])
    r2_mean = mean(r2_mean_list)
    MSE_mean = mean(MSE_mean_list)
    pearson_mean = mean(pearson_mean_list)
    return r2_mean, MSE_mean, pearson_mean

def rfrFeatSelect(X, y, test_size=0.1, random_state = 0, iterations = 1):
    r2_mean_list = []
    MSE_mean_list = []
    pearson_mean_list = []
    for i in range(iterations):
        print(f'Iteration: {i+1}')
        X_train, X_val, y_train, y_val = cell_line_split(X, y, test_size, random_state) #train-validation split
        classify = RandomForestRegressor(n_jobs=-1)
        classify.fit(X_train.values, y_train)
        y_pred = classify.predict(X_val.values)
        featSelect = classify.feature_importances_
        r2 = r2_score(y_val, y_pred)
        MSE = mean_squared_error(y_val, y_pred)
        pearson = pearsonr(y_val.values, y_pred)
        r2_mean_list.append(r2)
        MSE_mean_list.append(MSE)
        pearson_mean_list.append(pearson[0])
    r2_mean = mean(r2_mean_list)
    MSE_mean = mean(MSE_mean_list)
    pearson_mean = mean(pearson_mean_list)
    return r2_mean, MSE_mean, pearson_mean, featSelect
